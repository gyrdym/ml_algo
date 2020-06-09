import 'package:ml_algo/src/common/exception/invalid_class_labels_exception.dart';

void validateClassLabels(num positiveLabel, num negativeLabel) {
  if (positiveLabel == negativeLabel) {
    throw InvalidClassLabelsException(positiveLabel, negativeLabel);
  }
}
