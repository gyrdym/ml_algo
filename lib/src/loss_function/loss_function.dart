library loss_function;

import 'dart:math' as math;
import 'package:dart_ml/src/math/vector/vector.dart';

part 'cross_entropy.dart';
part 'squared_loss.dart';

abstract class LossFunction {
  double function(Vector w, Vector x, double y);

  factory LossFunction.Squared() => const _SquaredLoss();
  factory LossFunction.CrossEntropy() => const _CrossEntropy();
}