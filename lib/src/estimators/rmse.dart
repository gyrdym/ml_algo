import 'dart:math' as math;

import 'package:dart_ml/src/estimators/estimator.dart';
import 'package:dart_ml/src/math/vector_interface.dart';

class RMSEEstimator implements Estimator {
  double calculateError(VectorInterface predictedLabels, VectorInterface origLabels) =>
    math.sqrt(((predictedLabels - origLabels).intPow(2)).mean());
}
