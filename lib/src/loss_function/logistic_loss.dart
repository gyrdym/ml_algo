part of 'package:dart_ml/src/implementation.dart';

class _LogisticLoss implements LossFunction {
  const _LogisticLoss();

  @override
  double loss(double predictedLabel, double originalLabel) =>
      log(1.0 + exp(-originalLabel * predictedLabel)) / LN2;
}