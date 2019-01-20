import 'package:ml_algo/src/data_preprocessing/ml_data/labels_extractor/labels_extractor.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/value_converter/value_converter.dart';

class MLDataLabelsExtractorImpl implements MLDataLabelsExtractor {
  final List<bool> readMask;
  final int labelIdx;
  final int rowsNum;
  final MLDataValueConverter valueConverter;

  MLDataLabelsExtractorImpl(this.readMask, this.labelIdx, this.valueConverter)
      : rowsNum = readMask.where((bool flag) => flag).length;

  @override
  List<double> extract(List<List<Object>> data) {
    final result = List<double>(rowsNum);
    int _i = 0;
    for (int i = 0; i < data.length; i++) {
      if (readMask[i] == true) {
        final dynamic rawValue = data[i][labelIdx];
        final convertedValue = valueConverter.convert(rawValue);
        result[_i++] = convertedValue;
      }
    }
    return result;
  }
}
