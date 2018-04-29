import 'dart:math' as math;

import 'package:dart_ml/src/cost_function/cost_function.dart';
import 'package:simd_vector/vector.dart';

class LogLikelihoodCost implements CostFunction {
  const LogLikelihoodCost();

  @override
  double getCost(double score, double yOrig) {
    final probability = _sigmoidMap(score);
    return _indicator(yOrig, -1.0) * math.log(1 - probability) + _indicator(yOrig, 1.0) * math.log(probability);
  }

  @override
  double getPartialDerivative(
    int wIdx,
    covariant Float32x4Vector x,
    covariant Float32x4Vector w,
    double y
  ) =>  x[wIdx] * (_indicator(y, 1.0) - _sigmoidMap(x.dot(w)));

  int _indicator(double y, double sign) => sign == y ? 1 : 0;

  double _sigmoidMap(double score) => 1 / (1.0 + math.exp(-score));
}
