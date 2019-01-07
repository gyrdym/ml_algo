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
  final Map<int, CategoricalDataEncoder> _indexToEncoder = {};
  final Map<String, CategoricalDataEncoderType> _categoryNameToEncoderType;
  final Map<int, CategoricalDataEncoderType> _categoryIndexToEncoderType;
  final Map<String, List<Object>> _categories;

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
  }) :
        _csvCodec = CsvCodec(eol: eol),
        _file = File(fileName),
        _labelIdx = labelIdx,
        _headerExists = headerExists,
        _rows = rows,
        _columns = columns,
        _categories = categories,
        _categoryNameToEncoderType = categoryNameToEncoder,
        _categoryIndexToEncoderType = categoryIndexToEncoder,
        _encoderFactory = encoderFactory {

    _validateArgs(
      labelIdx,
      rows,
      columns,
    );
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

    if (_labelIdx >= _records.first.length || _labelIdx < 0) {
      throw RangeError.range(_labelIdx, 0, _records.first.length - 1, null,
          _wrapErrorMessage('Invalid label column number'));
    }

    _setEncoders(_records);

    _isCategoricalDataExist = _indexToEncoder.isNotEmpty;

    return data;
  }

  void _setEncoders(List<List<dynamic>> data) {
    if (_headerExists && _isCategoryNameToEncoderTypeDefined && !_isCategoryIndexToEncoderTypeDefined) {
      // create category index to encoder map from category name to encoder map
    }

    if (!_isCategoryIndexToEncoderTypeDefined && _categories.isNotEmpty) {
      // create fallback encoders
    }

    if (_isCategoryIndexToEncoderTypeDefined) {
      
    }
  }

  bool get _isCategoryNameToEncoderTypeDefined =>
      _categoryNameToEncoderType != null && _categoryNameToEncoderType.isNotEmpty;

  bool get _isCategoryIndexToEncoderTypeDefined =>
      _categoryIndexToEncoderType != null && _categoryIndexToEncoderType.isNotEmpty;

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

  void _validateArgs(int labelIdx, Iterable<Tuple2<int, int>> rows, Iterable<Tuple2<int, int>> columns) {
    final validators = [
      () => _validateLabelIdx(labelIdx),
      () => _validateReadRanges(rows),
      () => _validateReadRanges(columns, labelIdx),
    ];
    for (int i = 0; i < validators.length; i++) {
      final errorMsg = validators[i]();
      if (errorMsg != '') {
        throw Exception(errorMsg);
      }
    }
  }

  String _validateLabelIdx(int labelIdx) {
    if (labelIdx == null) {
      return _wrapErrorMessage('label index must not be null');
    }
    return '';
  }

  String _validateReadRanges(Iterable<Tuple2<int, int>> ranges, [int labelIdx]) {
    if (ranges == null) {
      return '';
    }

    String errorMessage = '';
    Tuple2<int, int> prevRange;
    bool isLabelInRanges = false;

    ranges.forEach((Tuple2<int, int> range) {
      if (range.item1 > range.item2) {
        errorMessage = _wrapErrorMessage('left boundary of the range $range is greater than the right one');
      }
      if (prevRange != null && prevRange.item2 >= range.item1) {
        errorMessage = _wrapErrorMessage('$prevRange and $range ranges are intersecting');
      }
      if (labelIdx != null && labelIdx >= range.item1 && labelIdx <= range.item2) {
        isLabelInRanges = true;
      }
      prevRange = range;
    });

    if (labelIdx != null && !isLabelInRanges) {
      errorMessage = _wrapErrorMessage('label index $_labelIdx is not in provided ranges $ranges');
    }

    return errorMessage;
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
