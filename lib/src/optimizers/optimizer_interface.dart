import 'package:dart_ml/src/math/vector/vector_interface.dart';

abstract class OptimizerInterface<T extends VectorInterface> {
  T optimize(List<T> features, List<double> labels);
}