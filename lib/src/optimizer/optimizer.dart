import 'package:dart_ml/src/math/vector/vector.dart';

abstract class Optimizer {
  Vector optimize(List<Vector> features, Vector labels, Vector weights);
}