import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:csv/csv.dart';
import 'package:ml_algo/categorical_data_encoder_type.dart';
import 'package:ml_algo/encode_unknown_value_strategy.dart';
import 'package:ml_algo/float32x4_csv_ml_data.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/one_hot_encoder.dart';
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

  static const String _errorPrefix = 'Csv ML Data';

  List<List<dynamic>> _data;
  List<List<dynamic>> _records;
  MLMatrix<Float32x4> _features;
  MLVector<Float32x4> _labels;
  List<String> _originalHeader;
  List<String> _header;
  CategoricalDataEncoder _categoricalEncoder;
  List<bool> _rowsMask;
  int _actualRowsNum;
  List<bool> _columnsMask;
  int _actualColumnsNum;

  Float32x4CsvMLDataInternal.fromFile(String fileName, {
    String eol = '\n',
    int labelIdx,
    bool headerExists = true,
    CategoricalDataEncoderType encoderType = CategoricalDataEncoderType.oneHot,
    EncodeUnknownValueStrategy encodeUnknownStrategy = EncodeUnknownValueStrategy.throwError,
    Map<String, List<Object>> categories,
    Map<int, List<Object>> categoriesByIndexes,
    Map<String, CategoricalDataEncoderType> categoryToEncoder,
    Map<int, CategoricalDataEncoderType> categoryIndexToEncoder,
    List<Tuple2<int, int>> rows,
    List<Tuple2<int, int>> columns,
    CategoricalDataEncoder categoricalEncoderFactory(),
  }) :
        _csvCodec = CsvCodec(eol: eol),
        _file = File(fileName),
        _labelIdx = labelIdx,
        _headerExists = headerExists,
        _rows = rows,
        _columns = columns {

    _validateArgs(
      labelIdx,
      rows,
      columns,
    );

    if (categories != null) {
      _categoricalEncoder = categoricalEncoderFactory != null
          ? categoricalEncoderFactory()
          : _createCategoricalDataEncoder(encoderType, categories, encodeUnknownStrategy);
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

    _originalHeader = _headerExists
        ? data[0].map((dynamic el) => el.toString()).toList(growable: true)
        : null;
    _records = _extractRecords(data);

    if (_labelIdx >= _records.first.length || _labelIdx < 0) {
      throw RangeError.range(_labelIdx, 0, _records.first.length - 1, null,
          _wrapErrorMessage('Invalid label column number'));
    }

    return data;
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
        features[_i++] = _categoricalEncoder != null
            ? _convertFeaturesWithCategoricalData(featuresRaw, labelIdx)
            : _convertFeatures(featuresRaw, labelIdx);
      }
    }
    return features;
  }

  List<double> _extractLabels(int labelPos) {
    final labelIdx = labelPos ?? _records.first.length - 1;
    final result = <double>[];
    for (int i = 0; i < _records.length; i++) {
      if (_rowsMask != null && _rowsMask[i] == false) {
        continue;
      }
      final dynamic rawValue = _records[i][labelIdx];
      final convertedValue = _convertValueToDouble(rawValue);
      result.add(convertedValue);
    }
    return result;
  }

  /// Light-weight method for data encoding without any checks if the current feature is categorical
  List<double> _convertFeatures(List<Object> features, int labelIdx) {
    final converted = <double>[];
    for (int i = 0; i < features.length; i++) {
      final feature = features[i];
      if (labelIdx == i || (_columnsMask != null && _columnsMask[i] == false)) {
        continue;
      }
      converted.add(_convertValueToDouble(feature));
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
      final columnTitle = _originalHeader[i];
      Iterable<double> expanded;
      if (_categoricalEncoder.categories.containsKey(columnTitle)) {
        expanded = _categoricalEncoder.encode(columnTitle, feature);
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

  CategoricalDataEncoder _createCategoricalDataEncoder(
      CategoricalDataEncoderType encoderType,
      Map<String, List<Object>> categoricalDataDescr,
      EncodeUnknownValueStrategy encodeUnknownStrategy,
  ) {
    switch (encoderType) {
      case CategoricalDataEncoderType.oneHot:
        return OneHotEncoder(categoricalDataDescr, encodeUnknownStrategy);
      default:
        throw UnsupportedError(_wrapErrorMessage('unsupported categorical categorical_encoder type $encoderType'));
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
