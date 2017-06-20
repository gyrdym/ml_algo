part of loss_function;

class _SquaredLoss implements LossFunction {
  const _SquaredLoss();

  @override
  double loss(double predictedLabel, double originalLabel) => math.pow(predictedLabel - originalLabel, 2);
}