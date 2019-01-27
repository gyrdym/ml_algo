import 'dart:math' as math;

import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_linalg/linalg.dart';

class SquaredCost<T> implements CostFunction<T> {
  @override
  double getCost(double predictedLabel, double originalLabel) => math.pow(predictedLabel - originalLabel, 2).toDouble();

  @override
  MLVector<T> getGradient(MLMatrix<T> x, MLVector<T> w, MLVector<T> y) =>
      (x.transpose() * -2 * (y - x * w)).toVector();

  @override
  double getSparseSolutionPartial(int wIdx, MLVector<T> x, MLVector<T> w, double y) =>
      x[wIdx] * (y - x.dot(w) + x[wIdx] * w[wIdx]);
}
