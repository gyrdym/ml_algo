import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:csv/csv.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/ml_data.dart';
import 'package:ml_linalg/float32x4_matrix.dart';
import 'package:ml_linalg/float32x4_vector.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class CsvMLData implements MLData<Float32x4> {
  final CsvCodec _csvCodec;
  final File _file;
  final int _labelPos;
  final bool _headerExists;

  Future<List<List<dynamic>>> _textTransform;
  List<List<dynamic>> _records;
  MLMatrix<Float32x4> _features;
  MLVector<Float32x4> _labels;

  CsvMLData(String fileName, {String eol = '\n', int labelPos, bool headerExists = true}) :
        _csvCodec = CsvCodec(eol: eol),
        _file = File(fileName),
        _labelPos = labelPos,
        _headerExists = headerExists {

    final fileStream = _file.openRead();
    _textTransform = (fileStream.transform(utf8.decoder).transform(_csvCodec.decoder).toList());
  }

  @override
  Future<MLMatrix<Float32x4>> get features async {
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

  Future<List<List<dynamic>>> _extractRecords() async => (await _textTransform).sublist(_headerExists ? 1 : 0);

  List<List<double>> _extractFeatures(int labelPos) {
    final lastIdx = _records.first.length - 1;
    final labelIdx = labelPos ?? lastIdx;
    return _records.map((List item) {
      final first = item.sublist(0, labelIdx);
      final last = labelIdx < lastIdx ? item.sublist(labelIdx + 1) : <Object>[];
      return _convertFeatures(first..addAll(last));
    }).toList(growable: false);
  }

  List<double> _extractLabels(int labelPos) {
    final labelIdx = labelPos ?? _records.first.length - 1;
    return _records.map((List<dynamic> item) => (item[labelIdx] as num).toDouble()).toList(growable: false);
  }

  List<double> _convertFeatures(List<Object> item) => item.map((Object feature) =>
      (feature as num).toDouble()).toList();
}
