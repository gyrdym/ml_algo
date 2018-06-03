import 'dart:math' as math;

import 'package:dart_ml/src/cost_function/cost_function.dart';
import 'package:linalg/vector.dart';

class SquaredCost implements CostFunction {
  const SquaredCost();

  @override
  double getCost(double predictedLabel, double originalLabel) => math.pow(predictedLabel - originalLabel, 2);

  @override
  double getPartialDerivative(
    int wIdx,
    covariant Float32x4Vector x,
    covariant Float32x4Vector w,
    double y
  ) => -2.0 * x[wIdx] * (y - x.dot(w));
}