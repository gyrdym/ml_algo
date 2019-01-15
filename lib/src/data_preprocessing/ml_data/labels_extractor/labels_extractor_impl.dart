import 'package:ml_algo/src/data_preprocessing/ml_data/labels_extractor/labels_extractor.dart';

class MLDataLabelsExtractorImpl implements MLDataLabelsExtractor {
  final List<bool> readMask;
  final int labelIdx;
  final int rowsNum;

  MLDataLabelsExtractorImpl(this.readMask, this.labelIdx)
      : rowsNum = readMask.where((bool flag) => flag).length;

  @override
  List<double> extract(List<List<Object>> data) {
    final result = List<double>(rowsNum);
    int _i = 0;
    for (int i = 0; i < data.length; i++) {
      if (readMask[i] == true) {
        final dynamic rawValue = data[i][labelIdx];
        final convertedValue = _convertValueToDouble(rawValue);
        result[_i++] = convertedValue;
      }
    }
    return result;
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
}
