import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_ml/src/cost_function/cost_function.dart';
import 'package:linalg/vector.dart';

class SquaredCost implements CostFunction<Float32x4> {
  const SquaredCost();

  @override
  double getCost(double predictedLabel, double originalLabel) =>
      math.pow(predictedLabel - originalLabel, 2).toDouble();

  @override
  double getPartialDerivative(int wIdx, Vector<Float32x4> x, Vector<Float32x4> w, double y) =>
      -2.0 * x[wIdx] * (y - x.dot(w));

  @override
  double getSparseSolutionPartial(int wIdx, Vector<Float32x4> x, Vector<Float32x4> w, double y) =>
      x[wIdx] * (y - x.dot(w) + x[wIdx] * w[wIdx]);
}