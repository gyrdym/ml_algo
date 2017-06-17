import 'dart:math' as math;
import 'package:dart_ml/src/math/vector/vector.dart';
import 'loss_function.dart';

class CrossEntropy implements LossFunction {
  const CrossEntropy();

  @override
  double function(Vector w, Vector x, double y) {
    double sigmoidValue = _sigmoid(w, x);
    return -(y * math.log(sigmoidValue) + (1 - y) * math.log(1 - sigmoidValue));
  }

  double _sigmoid(Vector w, Vector x) => math.exp(w.dot(x)) / (1 + math.exp(w.dot(x)));
}