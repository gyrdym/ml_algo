import 'dart:math' as math;

import 'package:dart_ml/src/estimators/estimator_interface.dart';
import 'package:dart_ml/src/math/vector/vector_interface.dart';

class RMSEEstimator implements EstimatorInterface {
  double calculateError(VectorInterface predictedLabels, VectorInterface origLabels) =>
    math.sqrt(((predictedLabels - origLabels).intPow(2)).mean());
}
