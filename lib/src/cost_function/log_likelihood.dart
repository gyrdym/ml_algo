import 'dart:math' as math;

import 'package:dart_ml/src/cost_function/cost_function.dart';

class LogLikelihoodCost implements CostFunction {
  const LogLikelihoodCost();

  @override
  double getCost(double score, double yOrig) {
    final probability = _sigmoidMap(score);
    return _indicator(yOrig, -1) * math.log(1 - probability) + _indicator(yOrig, 1) * math.log(probability);
  }

  int _indicator(double y, int sign) => sign == y ? 1 : 0;

  double _sigmoidMap(double score) => 1 / (1.0 + math.exp(-score));
}
