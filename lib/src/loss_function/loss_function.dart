import 'package:dart_vector/vector.dart' show Vector;

abstract class LossFunction {
  double function(Vector w, Vector x, double y);
}