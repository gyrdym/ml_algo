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
  int _actualRowsNum;
  List<bool> _columnsMask;
  int _actualColumnsNum;
  bool _isCategoricalDataExist = false;
  List<String> _originalHeader;

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
        _paramsValidator = paramsValidator {

    final errorMsg = _paramsValidator.validate(
      labelIdx: labelIdx,
      rows: rows,
      columns: columns,
      headerExists: headerExists,
      predefinedCategories: categories,
      nameToEncoder: categoryNameToEncoder,
      indexToEncoder: categoryIndexToEncoder,
    );
    if (errorMsg.isNotEmpty) {
      throw Exception(_wrapErrorMessage(errorMsg));
    }
  }

  @override
  Future<List<String>> get header async {
    _data ??= (await _prepareData(_rows, _columns));
    _header ??= _headerExists ? _extractHeader(_data) : null;
    return _header;
  }

  @override
  Future<MLMatrix<Float32x4>> get features async {
    _data ??= (await _prepareData(_rows, _columns));
    _features ??= Float32x4Matrix.from(_extractFeatures(_labelIdx));
    return _features;
  }

  @override
  Future<MLVector<Float32x4>> get labels async {
    _data ??= (await _prepareData(_rows, _columns));
    _labels ??= Float32x4Vector.from(_extractLabels(_labelIdx));
    return _labels;
  }

  Future<List<List<dynamic>>> _prepareData(Iterable<Tuple2<int, int>> rows, Iterable<Tuple2<int, int>> columns) async {
    final fileStream = _file.openRead();
    final data = await (fileStream.transform(utf8.decoder).transform(_csvCodec.decoder).toList());
    final rowsNum = data.length;
    final columnsNum = data.first.length;

    if (rows != null) {
      final rowsData = _createDataReadMask(rows, rowsNum);
      _rowsMask = rowsData.item1;
      _actualRowsNum = rowsData.item2;
    }

    if (columns != null) {
      final columnData = _createDataReadMask(columns, columnsNum);
      _columnsMask = columnData.item1;
      _actualColumnsNum = columnData.item2;
    }

    _actualRowsNum ??= rowsNum - (_headerExists ? 1 : 0);
    _actualColumnsNum ??= columnsNum;

    _records = _extractRecords(data);

    _originalHeader = _headerExists
        ? data[0].map((dynamic el) => el.toString()).toList(growable: true)
        : null;

    if (_labelIdx >= _records.first.length || _labelIdx < 0) {
      throw RangeError.range(_labelIdx, 0, _records.first.length - 1, null,
          _wrapErrorMessage('Invalid label column number'));
    }

    _setEncoders(_records);

    _isCategoricalDataExist = _indexToEncoder.isNotEmpty;

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

  List<String> _extractHeader(List<List<dynamic>> data) {
    final headerRow = data[0];
    final header = List<String>(_actualColumnsNum);
    int _i = 0;
    for (int i = 0; i < headerRow.length; i++) {
      if (_columnsMask == null || _columnsMask[i] == true) {
        header[_i] = headerRow[i].toString();
        _i++;
      }
    }
    return header;
  }

  List<List<dynamic>> _extractRecords(List<List<dynamic>> data) => data.sublist(_headerExists ? 1 : 0);

  List<List<double>> _extractFeatures(int labelPos) {
    final lastIdx = _records.first.length - 1;
    final labelIdx = labelPos ?? lastIdx;
    final features = List<List<double>>(_actualRowsNum);
    int _i = 0;
    for (int i = 0; i < _records.length; i++) {
      if (_rowsMask == null || _rowsMask[i] == true) {
        final featuresRaw = _records[i];
        features[_i++] = _isCategoricalDataExist
            ? _convertFeaturesWithCategoricalData(featuresRaw, labelIdx)
            : _convertFeatures(featuresRaw, labelIdx);
      }
    }
    return features;
  }

  List<double> _extractLabels(int labelPos) {
    final labelIdx = labelPos ?? _records.first.length - 1;
    final result = List<double>(_actualRowsNum);
    int _i = 0;
    for (int i = 0; i < _records.length; i++) {
      if (_rowsMask == null || _rowsMask[i] == true) {
        final dynamic rawValue = _records[i][labelIdx];
        final convertedValue = _convertValueToDouble(rawValue);
        result[_i++] = convertedValue;
      }
    }
    return result;
  }

  /// Light-weight method for data encoding without any checks if the current feature is categorical
  List<double> _convertFeatures(List<Object> features, int labelIdx) {
    final converted = List<double>(_actualColumnsNum - 1); // minus one column for label values
    int _i = 0;
    for (int i = 0; i < features.length; i++) {
      final feature = features[i];
      if (labelIdx != i && (_columnsMask == null || _columnsMask[i] == true)) {
        converted[_i++] = _convertValueToDouble(feature);
      }
    }
    return converted;
  }

  /// In order to avoid limitless checks if the current feature is categorical, let's create a separate method for
  /// data encoding if we know exactly that categories are presented in the data set
  List<double> _convertFeaturesWithCategoricalData(List<Object> features, int labelIdx) {
    final converted = <double>[];
    for (int i = 0; i < features.length; i++) {
      if (labelIdx == i || (_columnsMask != null  && _columnsMask[i] == false)) {
        continue;
      }
      final feature = features[i];
      Iterable<double> expanded;
      if (_indexToEncoder.containsKey(i)) {
        expanded = _indexToEncoder[i].encode(feature);
      } else {
        expanded = [_convertValueToDouble(feature)];
      }
      converted.addAll(expanded);
    }
    return converted;
  }

  double _convertValueToDouble(dynamic value) {
    if (value is String) {
      if (value.isEmpty) {
        return 0.0;
      } else {
        return double.parse(value);
      }
    } else {
      return (value as num).toDouble();
    }
  }

  Tuple2<List<bool>, int> _createDataReadMask(Iterable<Tuple2<int, int>> ranges, int limit) {
    final mask = List<bool>.filled(limit, false);
    int numOfElements = 0;
    ranges.take(limit).forEach((Tuple2<int, int> range) {
      if (range.item1 >= limit) {
        return false;
      }
      final end = math.min(limit, range.item2 + 1);
      mask.fillRange(range.item1, end, true);
      numOfElements += end - range.item1;
    });
    return Tuple2<List<bool>, int>(mask, numOfElements);
  }

  String _wrapErrorMessage(String text) => '$_errorPrefix: $text';
}
