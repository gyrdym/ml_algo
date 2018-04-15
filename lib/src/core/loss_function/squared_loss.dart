import 'dart:math' as math;

import 'package:dart_ml/src/core/loss_function/loss_function.dart';

class SquaredLoss implements LossFunction {
  const SquaredLoss();

  @override
  double loss(double predictedLabel, double originalLabel) => math.pow(predictedLabel - originalLabel, 2);
}