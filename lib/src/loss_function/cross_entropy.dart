part of 'package:dart_ml/src/implementation.dart';

class _CrossEntropy implements LossFunction {
  const _CrossEntropy();

  @override
  double loss(double predictedLabel, double originalLabel) {
    double sigmoidValue = _sigmoid(predictedLabel);
    return -(originalLabel * log(sigmoidValue) + (1 - originalLabel) * log(1 - sigmoidValue));
  }

  double _sigmoid(double z) => exp(z) / (1 + exp(z));
}
