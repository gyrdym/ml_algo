import 'dart:math' as math;

import 'package:dart_ml/src/estimator/estimator.dart';
import 'package:dart_ml/src/math/vector/vector.dart';

class RMSEEstimator implements Estimator {
  double calculateError(Vector predictedLabels, Vector origLabels) =>
    math.sqrt(((predictedLabels - origLabels).intPow(2)).mean());
}
