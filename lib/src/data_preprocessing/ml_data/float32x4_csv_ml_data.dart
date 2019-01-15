import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:csv/csv.dart';
import 'package:ml_algo/float32x4_csv_ml_data.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encode_unknown_strategy_type.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_factory.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_type.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/features_extractor/features_extractor.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/features_extractor/features_extractor_factory.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/features_extractor/features_extractor_factory_impl.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/header_extractor/header_extractor.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/header_extractor/header_extractor_factory.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/header_extractor/header_extractor_factory_impl.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/labels_extractor/labels_extractor.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/labels_extractor/labels_extractor_factory.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/labels_extractor/labels_extractor_factory_impl.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/validator/ml_data_params_validator.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/validator/ml_data_params_validator_impl.dart';
import 'package:ml_linalg/float32x4_matrix.dart';
import 'package:ml_linalg/float32x4_vector.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:tuple/tuple.dart';

class Float32x4CsvMLDataInternal implements Float32x4CsvMLData {
  final CsvCodec _csvCodec;
  final File _file;
  final int _labelIdx;
  final bool _headerExists;
  final List<Tuple2<int, int>> _rows;
  final List<Tuple2<int, int>> _columns;
  final CategoricalDataEncoderFactory _encoderFactory;
  final MLDataParamsValidator _paramsValidator;
  final MLDataHeaderExtractorFactory _headerExtractorFactory;
  final MLDataFeaturesExtractorFactory _featuresExtractorFactory;
  final MLDataLabelsExtractorFactory _labelsExtractorFactory;
  final Map<int, CategoricalDataEncoder> _indexToEncoder = {};
  final Map<String, CategoricalDataEncoderType> _nameToEncoderType;
  final Map<int, CategoricalDataEncoderType> _indexToEncoderType;
  final Map<String, List<Object>> _categories;
  final CategoricalDataEncoderType _fallbackEncoderType;

  static const String _errorPrefix = 'Csv ML Data';

  List<List<dynamic>> _data;
  List<List<dynamic>> _records;
  MLMatrix<Float32x4> _features;
  MLVector<Float32x4> _labels;
  List<String> _header;
  List<bool> _rowsMask;
  List<bool> _columnsMask;
  bool categoricalDataExists = false;
  List<String> _originalHeader;
  MLDataHeaderExtractor _headerExtractor;
  MLDataFeaturesExtractor _featuresExtractor;
  MLDataLabelsExtractor _labelsExtractor;

  Float32x4CsvMLDataInternal.fromFile(String fileName, {
    // public parameters
    String eol = '\n',
    int labelIdx,
    bool headerExists = true,
    CategoricalDataEncoderType encoderType = CategoricalDataEncoderType.oneHot,
    EncodeUnknownValueStrategy encodeUnknownStrategy = EncodeUnknownValueStrategy.throwError,
    Map<String, List<Object>> categories,
    Map<int, List<Object>> categoriesByIndexes,
    Map<String, CategoricalDataEncoderType> categoryNameToEncoder,
    Map<int, CategoricalDataEncoderType> categoryIndexToEncoder,
    List<Tuple2<int, int>> rows,
    List<Tuple2<int, int>> columns,

    // private parameters, they are hidden by the factory
    CategoricalDataEncoderFactory encoderFactory = const CategoricalDataEncoderFactory(),
    MLDataParamsValidator paramsValidator = const MLDataParamsValidatorImpl(),
    MLDataHeaderExtractorFactory headerExtractorFactory = const MLDataHeaderExtractorFactoryImpl(),
    MLDataFeaturesExtractorFactory featuresExtractorFactory = const MLDataFeaturesExtractorFactoryImpl(),
    MLDataLabelsExtractorFactory labelsExtractorFactory = const MLDataLabelsExtractorFactoryImpl(),
  }) :
        _csvCodec = CsvCodec(eol: eol),
        _file = File(fileName),
        _labelIdx = labelIdx,
        _headerExists = headerExists,
        _rows = rows,
        _columns = columns,
        _categories = categories,
        _nameToEncoderType = categoryNameToEncoder,
        _indexToEncoderType = categoryIndexToEncoder,
        _encoderFactory = encoderFactory,
        _fallbackEncoderType = encoderType,
        _paramsValidator = paramsValidator,
        _headerExtractorFactory = headerExtractorFactory,
        _featuresExtractorFactory = featuresExtractorFactory,
        _labelsExtractorFactory = labelsExtractorFactory {

    final errorMsg = _paramsValidator.validate(
      labelIdx: labelIdx,
      rows: rows,
      columns: columns,
      headerExists: headerExists,
      predefinedCategories: categories,
      namesToEncoders: categoryNameToEncoder,
      indexToEncoder: categoryIndexToEncoder,
    );
    if (errorMsg.isNotEmpty) {
      throw Exception(_wrapErrorMessage(errorMsg));
    }
  }

  @override
  Future<List<String>> get header async {
    _data ??= (await _prepareData(_rows, _columns));
    _header ??= _headerExists ? _headerExtractor.extract(_data) : null;
    return _header;
  }

  @override
  Future<MLMatrix<Float32x4>> get features async {
    _data ??= (await _prepareData(_rows, _columns));
    _features ??= Float32x4Matrix.from(_featuresExtractor.extract(_records, hasCategoricalData: categoricalDataExists));
    return _features;
  }

  @override
  Future<MLVector<Float32x4>> get labels async {
    _data ??= (await _prepareData(_rows, _columns));
    _labels ??= Float32x4Vector.from(_labelsExtractor.extract(_records));
    return _labels;
  }

  Future<List<List<dynamic>>> _prepareData([Iterable<Tuple2<int, int>> rows, Iterable<Tuple2<int, int>> columns]) async {
    final fileStream = _file.openRead();
    final data = await (fileStream.transform(utf8.decoder).transform(_csvCodec.decoder).toList());
    final rowsNum = data.length;
    final columnsNum = data.first.length;
    final _rowsMask = _createDataReadMask(rows ?? [Tuple2<int, int>(0, rowsNum - (_headerExists ? 2 : 1))], rowsNum);
    final _columnsMask = _createDataReadMask(columns ?? [Tuple2<int, int>(0, columnsNum - 1)], columnsNum);

    _records = _extractRecords(data);

    _originalHeader = _headerExists
        ? data[0].map((dynamic el) => el.toString()).toList(growable: true)
        : null;

    if (_labelIdx >= _records.first.length || _labelIdx < 0) {
      throw RangeError.range(_labelIdx, 0, _records.first.length - 1, null,
          _wrapErrorMessage('Invalid label column number'));
    }

    _setEncoders(_records);

    _headerExtractor = _headerExtractorFactory.create(_columnsMask);
    _featuresExtractor = _featuresExtractorFactory.create(_rowsMask, _columnsMask, _indexToEncoder, _labelIdx);
    _labelsExtractor = _labelsExtractorFactory.create(_rowsMask, _labelIdx);

    categoricalDataExists = _indexToEncoder.isNotEmpty;

    return data;
  }

  void _setEncoders(List<List<dynamic>> data) {
    final _isCategoryNameToEncoderTypeDefined =
        _nameToEncoderType != null && _nameToEncoderType.isNotEmpty;

    final _isCategoryIndexToEncoderTypeDefined =
      _indexToEncoderType != null && _indexToEncoderType.isNotEmpty;

    // let's cast all categorical data to column index-based format

    // create _categoryIndexToEncoderType from category names
    if (_headerExists && _isCategoryNameToEncoderTypeDefined && !_isCategoryIndexToEncoderTypeDefined) {
      _fillIndexToEncoderTypeMap();
    }

    // create map "column index" -> "encoder" from fully-predefined categories data - _categories
    if (!_isCategoryIndexToEncoderTypeDefined && _categories?.isNotEmpty == true) {
      _fillIndexToEncoderMapFromCategories();
    // create map "column index" -> "encoder" from map "column index" -> "encoder type"
    } else if (_isCategoryIndexToEncoderTypeDefined) {
      _fillIndexToEncoderMapFromEncoderTypes();
    }

    _setCategoryValues();
  }

  void _fillIndexToEncoderTypeMap() {
    for (int i = 0; i < _originalHeader.length; i++) {
      final name = _originalHeader[i];
      if (_nameToEncoderType.containsKey(name)) {
        _indexToEncoderType.putIfAbsent(i, () => _nameToEncoderType[name]);
      }
    }
  }

  void _fillIndexToEncoderMapFromCategories() {
    for (int i = 0; i < _originalHeader.length; i++) {
      final name = _originalHeader[i];
      if (_categories.containsKey(name)) {
        _indexToEncoder[i] = _encoderFactory.fromType(_fallbackEncoderType);
      }
    }
  }

  void _fillIndexToEncoderMapFromEncoderTypes() {
    for (int i = 0; i < _originalHeader.length; i++) {
      if (_indexToEncoderType.containsKey(i)) {
        final encoderType = _indexToEncoderType[i];
        _indexToEncoder[i] = _encoderFactory.fromType(encoderType);
      }
    }
  }

  void _setCategoryValues() {
    final values = <int, List<Object>>{};
    for (int i = 0; i < _records.length; i++) {
      for (int j = 0; j < _records[i].length; j++) {
        if (_indexToEncoder.containsKey(j)) {
          values.putIfAbsent(j, () => List<Object>(_records.length));
          values[j][i] = _records[i][j];
        }
      }
    }
    values.forEach((int column, List<Object> values) {
      _indexToEncoder[column].setCategoryValues(values);
    });
  }

  List<List<dynamic>> _extractRecords(List<List<dynamic>> data) => data.sublist(_headerExists ? 1 : 0);

  List<bool> _createDataReadMask(Iterable<Tuple2<int, int>> ranges, int limit) {
    final mask = List<bool>.filled(limit, false);
    ranges.take(limit).forEach((Tuple2<int, int> range) {
      if (range.item1 >= limit) {
        return false;
      }
      final end = math.min(limit, range.item2 + 1);
      mask.fillRange(range.item1, end, true);
    });
    return mask;
  }

  String _wrapErrorMessage(String text) => '$_errorPrefix: $text';
}
