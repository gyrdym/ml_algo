part of loss_function;

class _LogisticLoss implements LossFunction {
  const _LogisticLoss();

  @override
  double loss(double predictedLabel, double originalLabel) =>
      log2(1.0 + math.exp(-originalLabel * predictedLabel));

  double log2(double x) => math.log(x) / math.log(2);
}