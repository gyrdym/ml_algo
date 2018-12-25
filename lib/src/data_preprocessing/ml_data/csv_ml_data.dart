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

class Float32x4CsvMLDataInternal implements MLData<Float32x4> {
  final CsvCodec _csvCodec;
  final File _file;
  final int _labelPos;
  final bool _headerExists;

  Future<List<List<dynamic>>> _textTransform;
  List<List<dynamic>> _records;
  MLMatrix<Float32x4> _features;
  MLVector<Float32x4> _labels;
  List<String> _header;
  CategoricalDataEncoder _categoricalEncoder;

  Float32x4CsvMLDataInternal.fromFile(String fileName, {
    String eol = '\n',
    int labelPos,
    bool headerExists = true,
    CategoricalDataEncoderType encoderType = CategoricalDataEncoderType.oneHot,
    Map<String, List<Object>> categoricalDataDescr,
    CategoricalDataEncoder categoricalEncoderFactory(Map<String, List<Object>> dataDesrc),
  }) :
        _csvCodec = CsvCodec(eol: eol),
        _file = File(fileName),
        _labelPos = labelPos,
        _headerExists = headerExists {

    final fileStream = _file.openRead();
    _textTransform = (fileStream.transform(utf8.decoder).transform(_csvCodec.decoder).toList());

    if (categoricalDataDescr != null) {
      _categoricalEncoder = categoricalEncoderFactory != null
          ? categoricalEncoderFactory(categoricalDataDescr)
          : _createCategoricalDataEncoder(encoderType, categoricalDataDescr);
    }
  }

  @override
  Future<List<String>> get header async {
    if (!_headerExists) {
      return null;
    }
    _header ??= await _extractHeader();
    return _header;
  }

  @override
  Future<MLMatrix<Float32x4>> get features async {
    _header ??= await _extractHeader();
    await _prepareToRead();
    _features ??= Float32x4Matrix.from(_extractFeatures(_labelPos));
    return _features;
  }

  @override
  Future<MLVector<Float32x4>> get labels async {
    await _prepareToRead();
    _labels ??= Float32x4Vector.from(_extractLabels(_labelPos));
    return _labels;
  }

  Future _prepareToRead() async {
    _records ??= await _extractRecords();
    if (_labelPos != null && (_labelPos >= _records.first.length || _labelPos < 0)) {
      throw RangeError.range(_labelPos, 0, _records.first.length - 1, null, 'Invalid label column position');
    }
  }

  Future<List<String>> _extractHeader() async => (await _textTransform)[0]
      .map((dynamic label) => label.toString())
      .toList(growable: false);

  Future<List<List<dynamic>>> _extractRecords() async => (await _textTransform).sublist(_headerExists ? 1 : 0);

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
        throw UnsupportedError('CSV data: unsupported categorical categorical_encoder type $encoderType');
    }
  }
}
