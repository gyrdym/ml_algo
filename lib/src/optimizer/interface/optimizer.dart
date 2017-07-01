import 'package:dart_vector/vector.dart' show Vector;

abstract class Optimizer {
  Vector optimize(List<Vector> features, Vector labels, {Vector weights});
}