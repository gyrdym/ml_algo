import 'dart:math' as math;

import 'package:dart_ml/src/cost_function/cost_function.dart';

class SquaredCost implements CostFunction {
  const SquaredCost();

  @override
  double getCost(double predictedLabel, double originalLabel) => math.pow(predictedLabel - originalLabel, 2);
}