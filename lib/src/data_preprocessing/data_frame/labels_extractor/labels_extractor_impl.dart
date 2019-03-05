import 'package:logging/logging.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/labels_extractor/labels_extractor.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/value_converter/value_converter.dart';
import 'package:ml_algo/src/utils/logger/logger_mixin.dart';

class MLDataLabelsExtractorImpl with LoggerMixin
    implements MLDataLabelsExtractor {

  MLDataLabelsExtractorImpl(this.records, this.readMask, this.labelIdx,
      this.valueConverter, this.encoders, this.logger)
      : rowsNum = readMask.where((bool flag) => flag).length {
    if (readMask.length > records.length) {
      throwException(wrongReadMaskLengthMsg);
    }
    if (labelIdx >= records.first.length) {
      throwException(wrongLabelIndexMsg);
    }
  }

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
  final Map<int, CategoricalDataEncoder> encoders;
  final MLDataValueConverter valueConverter;

  @override
  final Logger logger;

  @override
  List<List<double>> getLabels() {
    final result = List<List<double>>(rowsNum);
    int _i = 0;
    final categoricalDataExist = encoders != null &&
        encoders.containsKey(labelIdx);
    for (int row = 0; row < readMask.length; row++) {
      if (readMask[row] == true) {
        final dynamic rawValue = records[row][labelIdx];
        final convertedValue = categoricalDataExist
            ? encoders[labelIdx].encodeSingle(rawValue).toList(growable: false)
            : [valueConverter.convert(rawValue)];
        result[_i++] = convertedValue;
      }
    }
    return result;
  }
}
