import 'dart:math' as math;
import 'package:dart_vector/vector.dart' show Vector;
import 'loss_function.dart';

class SquaredLoss implements LossFunction {
  const SquaredLoss();

  @override
  double function(Vector w, Vector x, double y) => math.pow(w.dot(x) - y, 2);
}