import 'package:dart_ml/src/math/vector/vector.dart';

abstract class LossFunction {
  double function(Vector w, Vector x, double y);
}