import 'dart:math' as math;

import 'package:dart_ml/src/estimator/estimator.dart';
import 'package:dart_vector/vector.dart';

class RMSEEstimator implements Estimator {
  double calculateError(Float32x4Vector predictedLabels, Float32x4Vector origLabels) =>
    math.sqrt(((predictedLabels - origLabels).intPow(2)).mean());
}
