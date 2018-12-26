import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:csv/csv.dart';
import 'package:ml_algo/categorical_data_encoder_type.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/one_hot_encoder.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/ml_data.dart';
import 'package:ml_linalg/float32x4_matrix.dart';
import 'package:ml_linalg/float32x4_vector.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:tuple/tuple.dart';

class Float32x4CsvMLDataInternal implements MLData<Float32x4> {
  final CsvCodec _csvCodec;
  final File _file;
  final int _labelPos;
  final bool _headerExists;
  final _dataReadyCompleter = Completer<Null>();

  static const String _errorPrefix = 'Csv ML Data';

  List<List<dynamic>> _records;
  MLMatrix<Float32x4> _features;
  MLVector<Float32x4> _labels;
  List<String> _header;
  CategoricalDataEncoder _categoricalEncoder;
  List<bool> _columnsToReadMask;

  Float32x4CsvMLDataInternal.fromFile(String fileName, {
    String eol = '\n',
    int labelPos,
    bool headerExists = true,
    CategoricalDataEncoderType encoderType = CategoricalDataEncoderType.oneHot,
    Map<String, List<Object>> categories,
    List<Tuple2<int, int>> columnsToRead,
    CategoricalDataEncoder categoricalEncoderFactory(Map<String, Iterable<Object>> dataDesrc),
  }) :
        _csvCodec = CsvCodec(eol: eol),
        _file = File(fileName),
        _labelPos = labelPos,
        _headerExists = headerExists {

    if (categories != null) {
      _categoricalEncoder = categoricalEncoderFactory != null
          ? categoricalEncoderFactory(categories)
          : _createCategoricalDataEncoder(encoderType, categories);
    }

    _prepareData(columnsToRead);
  }

  Future get _dataReadiness => _dataReadyCompleter.future;

  @override
  Future<List<String>> get header async {
    await _dataReadiness;
    return _header;
  }

  @override
  Future<MLMatrix<Float32x4>> get features async {
    await _dataReadiness;
    _features ??= Float32x4Matrix.from(_extractFeatures(_labelPos));
    return _features;
  }

  @override
  Future<MLVector<Float32x4>> get labels async {
    await _dataReadiness;
    _labels ??= Float32x4Vector.from(_extractLabels(_labelPos));
    return _labels;
  }

  Future _prepareData(Iterable<Tuple2<int, int>> columnsToRead) async {
    final fileStream = _file.openRead();
    final data = await (fileStream.transform(utf8.decoder).transform(_csvCodec.decoder).toList());
    final columnsNum = data.first.length;

    if (columnsToRead != null) {
      _columnsToReadMask = _createColumnsToReadMask(columnsToRead, columnsNum);
    }

    _header = _headerExists ? _extractHeader(data) : null;
    _records = _extractRecords(data);

    if (_labelPos != null && (_labelPos >= _records.first.length || _labelPos < 0)) {
      throw RangeError.range(_labelPos, 0, _records.first.length - 1, null,
          getErrorMessage('Invalid label column number'));
    }

    _dataReadyCompleter.complete();
  }

  List<String> _extractHeader(List<List<dynamic>> data) {
    final headerRaw = data[0];
    // @TODO: replace with a fixed-length list
    final header = <String>[];
    for (int i = 0; i < headerRaw.length; i++) {
      if (_columnsToReadMask == null || _columnsToReadMask[i] == true) {
        header.add(headerRaw[i].toString());
      }
    }
    return header;
  }

  List<List<dynamic>> _extractRecords(List<List<dynamic>> data) => data.sublist(_headerExists ? 1 : 0);

  List<List<double>> _extractFeatures(int labelPos) {
    final lastIdx = _records.first.length - 1;
    final labelIdx = labelPos ?? lastIdx;
    return _records.map((List item) {
      final first = item.sublist(0, labelIdx);
      final last = labelIdx < lastIdx ? item.sublist(labelIdx + 1) : <Object>[];
      final featuresWithoutLabels = first..addAll(last);
      if (_categoricalEncoder != null) {
        return _convertFeaturesWithCategoricalData(featuresWithoutLabels);
      } else {
        return _convertFeatures(featuresWithoutLabels);
      }
    }).toList(growable: false);
  }

  List<double> _extractLabels(int labelPos) {
    final labelIdx = labelPos ?? _records.first.length - 1;
    return _records.map((List<dynamic> item) => (item[labelIdx] as num).toDouble()).toList(growable: false);
  }

  /// Light-weight method for data encoding without any checks if the current feature is categorical
  List<double> _convertFeatures(List<Object> item) => item.map((Object feature) =>
      (feature as num).toDouble()).toList();

  /// In order to avoid limitless checks if the current feature is categorical, let's create a separate method for
  /// data encoding if we know exactly that categories are presented in the data set
  List<double> _convertFeaturesWithCategoricalData(List<Object> item) {
    int columnNum = 0;
    return item.expand((Object feature) {
      final columnTitle = _header[columnNum];
      Iterable<double> expanded;
      if (_categoricalEncoder.categories.containsKey(columnTitle)) {
        expanded = _categoricalEncoder.encode(columnTitle, feature);
      } else {
        expanded = [(feature as num).toDouble()];
      }
      columnNum++;
      return expanded;
    }).toList();
  }

  CategoricalDataEncoder _createCategoricalDataEncoder(
      CategoricalDataEncoderType encoderType,
      Map<String, List<Object>> categoricalDataDescr,
  ) {
    switch (encoderType) {
      case CategoricalDataEncoderType.oneHot:
        return OneHotEncoder(categoricalDataDescr);
      default:
        throw UnsupportedError(getErrorMessage('unsupported categorical categorical_encoder type $encoderType'));
    }
  }

  List<bool> _createColumnsToReadMask(Iterable<Tuple2<int, int>> ranges, int columnsNum) {
    if (ranges.length > columnsNum) {
      throw Exception(getErrorMessage('too many column ranges are provided'));
    }

    final mask = List<bool>.filled(columnsNum, false);
    Tuple2<int, int> prevRange;

    ranges.forEach((Tuple2<int, int> range) {
      if (range.item1 >= columnsNum || range.item2 >= columnsNum) {
        throw Exception(getErrorMessage('one of the range $range\'s boundaries is greater than the total dataset columns'
            ' number ($columnsNum)'));
      }
      if (range.item1 > range.item2) {
        throw Exception(getErrorMessage('left boundary of the range $range is greater than the right one'));
      }
      if (prevRange != null && prevRange.item2 >= range.item1) {
        throw Exception(getErrorMessage('$prevRange and $range ranges are intersecting'));
      }
      final rangeLength = range.item2 - range.item1;
      mask.setRange(range.item1, range.item2, List<bool>.filled(rangeLength, true));
      prevRange = range;
    });

    return mask;
  }

  String getErrorMessage(String text) => '$_errorPrefix: $text';
}
