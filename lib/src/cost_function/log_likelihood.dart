import 'dart:math' as math;

import 'package:dart_ml/src/cost_function/cost_function.dart';
import 'package:simd_vector/vector.dart';

class LogLikelihoodCost implements CostFunction {
  const LogLikelihoodCost();

  @override
  double getCost(double score, double yOrig) {
    final probability = linkScoreToProbability(score);
    return _indicator(yOrig, -1.0) * math.log(1 - probability) + _indicator(yOrig, 1.0) * math.log(probability);
  }

  @override
  double getPartialDerivative(
    int wIdx,
    covariant Float64x2Vector x,
    covariant Float64x2Vector w,
    double y
  ) =>  x[wIdx] * (_indicator(y, 1.0) - linkScoreToProbability(x.dot(w)));

  int _indicator(double y, double target) => target == y ? 1 : 0;

  @override
  double linkScoreToProbability(double score) => 1 / (1.0 + math.exp(-score));
}
