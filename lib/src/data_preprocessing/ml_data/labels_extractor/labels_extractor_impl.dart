import 'package:logging/logging.dart';
import 'package:ml_algo/src/utils/logger/logger_mixin.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/labels_extractor/labels_extractor.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/value_converter/value_converter.dart';

class MLDataLabelsExtractorImpl extends Object
    with LoggerMixin
    implements MLDataLabelsExtractor {
  static const String wrongReadMaskLengthMsg =
      'Rows read mask for label column should not be greater than the number '
      'of labels in the column!';

  static const String wrongLabelIndexMsg =
      'Labels column index should be less than actual columns number of the '
      'dataset!';

  final List<List<Object>> records;
  final List<bool> readMask;
  final int labelIdx;
  final int rowsNum;
  final MLDataValueConverter valueConverter;

  @override
  final Logger logger;

  MLDataLabelsExtractorImpl(this.records, this.readMask, this.labelIdx,
      this.valueConverter, this.logger)
      : rowsNum = readMask.where((bool flag) => flag).length {
    if (readMask.length > records.length) {
      throwException(wrongReadMaskLengthMsg);
    }
    if (labelIdx >= records.first.length) {
      throwException(wrongLabelIndexMsg);
    }
  }

  @override
  List<double> getLabels() {
    final result = List<double>(rowsNum);
    int _i = 0;
    for (int i = 0; i < readMask.length; i++) {
      if (readMask[i] == true) {
        final dynamic rawValue = records[i][labelIdx];
        final convertedValue = valueConverter.convert(rawValue);
        result[_i++] = convertedValue;
      }
    }
    return result;
  }
}
