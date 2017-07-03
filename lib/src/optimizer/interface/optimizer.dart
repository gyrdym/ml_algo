import 'package:dart_vector/vector.dart';

abstract class Optimizer {
  Float32x4Vector optimize(List<Float32x4Vector> features, List<double> labels, {Float32x4Vector weights});
}