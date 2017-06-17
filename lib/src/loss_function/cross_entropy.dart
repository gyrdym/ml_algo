part of loss_function;

class _CrossEntropy implements LossFunction {
  const _CrossEntropy();

  @override
  double function(Vector w, Vector x, double y) {
    double sigmoidValue = _sigmoid(w, x);
    return -(y * math.log(sigmoidValue) + (1 - y) * math.log(1 - sigmoidValue));
  }

  double _sigmoid(Vector w, Vector x) => math.exp(w.dot(x)) / (1 + math.exp(w.dot(x)));
}