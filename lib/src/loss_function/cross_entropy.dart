part of loss_function;

class _CrossEntropy implements LossFunction {
  const _CrossEntropy();

  @override
  double loss(double predictedLabel, double originalLabel) {
    double sigmoidValue = _sigmoid(predictedLabel);
    return -(originalLabel * math.log(sigmoidValue) + (1 - originalLabel) * math.log(1 - sigmoidValue));
  }

  double _sigmoid(double z) => math.exp(z) / (1 + math.exp(z));
}
