library loss_function;

import 'dart:math' as math;

part 'cross_entropy.dart';
part 'squared_loss.dart';
part 'logistic_loss.dart';

abstract class LossFunction {
  double loss(double predictedLabel, double originalLabel);

  factory LossFunction.Squared() => const _SquaredLoss();
  factory LossFunction.CrossEntropy() => const _CrossEntropy();
  factory LossFunction.LogisticLoss() => const _LogisticLoss();
}