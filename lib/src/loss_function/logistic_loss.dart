part of loss_function;

class _LogisticLoss implements LossFunction {
  const _LogisticLoss();

  @override
  double loss(double predictedLabel, double originalLabel) =>
      math.log(1.0 + math.exp(-originalLabel * predictedLabel)) / math.LN2;
}