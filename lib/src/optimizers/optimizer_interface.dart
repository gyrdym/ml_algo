import 'package:dart_ml/src/enums.dart';
import 'package:dart_ml/src/math/vector_interface.dart';

abstract class OptimizerInterface<T extends VectorInterface> {
  T optimize(List<T> features, T labels, CostFunction metric);
}
