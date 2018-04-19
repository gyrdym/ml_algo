import 'dart:math' as math;

import 'package:dart_ml/src/loss_function/loss_function.dart';

class CrossEntropyLoss implements LossFunction {
  const CrossEntropyLoss();

  @override
  double loss(double predictedLabel, double originalLabel) {
    double sigmoidValue = _sigmoid(predictedLabel);
    return -(originalLabel * math.log(sigmoidValue) + (1 - originalLabel) * math.log(1 - sigmoidValue));
  }

  double _sigmoid(double z) => math.exp(z) / (1 + math.exp(z));
}
