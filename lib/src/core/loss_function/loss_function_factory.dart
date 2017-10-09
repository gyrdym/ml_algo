part of 'package:dart_ml/src/core/implementation.dart';

class LossFunctionFactory {
  static LossFunction Squared() => const _SquaredLoss();
  static LossFunction CrossEntropy() => const _CrossEntropy();
  static LossFunction LogisticLoss() => const _LogisticLoss();
}