part of loss_function;

class _SquaredLoss implements LossFunction {
  const _SquaredLoss();

  @override
  double function(Vector w, Vector x, double y) => math.pow(w.dot(x) - y, 2);
}