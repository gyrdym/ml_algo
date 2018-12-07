import 'dart:math' as math;
import 'dart:typed_data';

import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_linalg/linalg.dart';

class SquaredCost implements CostFunction<Float32x4> {
  const SquaredCost();

  @override
  double getCost(double predictedLabel, double originalLabel) => math.pow(predictedLabel - originalLabel, 2).toDouble();

  @override
  double getPartialDerivative(int wIdx, MLVector<Float32x4> x, MLVector<Float32x4> w, double y) =>
      -2.0 * x[wIdx] * (y - x.dot(w));

  @override
  MLMatrix<Float32x4> getGradient(MLMatrix<Float32x4> x, MLMatrix<Float32x4> w, MLMatrix<Float32x4> y) =>
      x.transpose() * -2 * (y - x * w);

  @override
  double getSparseSolutionPartial(int wIdx, MLVector<Float32x4> x, MLVector<Float32x4> w, double y) =>
      x[wIdx] * (y - x.dot(w) + x[wIdx] * w[wIdx]);
}
