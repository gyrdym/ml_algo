import 'package:ml_algo/src/common/exception/invalid_probability_threshold_exception.dart';

void validateProbabilityThreshold(num threshold) {
  if (threshold <= 0.0 || threshold >= 1.0) {
    throw InvalidProbabilityThresholdException(threshold);
  }
}
