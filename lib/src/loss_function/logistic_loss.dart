import 'dart:math' as math;

import 'package:dart_ml/src/loss_function/loss_function.dart';

class LogisticLoss implements LossFunction {
  const LogisticLoss();

  @override
  double loss(double predictedLabel, double originalLabel) =>
      math.log(1.0 + math.exp(-originalLabel * predictedLabel)) / math.LN2;
}