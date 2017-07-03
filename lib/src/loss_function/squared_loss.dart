import 'dart:math' as math;
import 'package:dart_vector/vector.dart';
import 'loss_function.dart';

class SquaredLoss implements LossFunction {
  const SquaredLoss();

  @override
  double function(Float32x4Vector w, Float32x4Vector x, double y) => math.pow(w.dot(x) - y, 2);
}