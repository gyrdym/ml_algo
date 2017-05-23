import 'package:dart_ml/src/math/vector/vector_interface.dart';

abstract class OptimizerInterface {
  VectorInterface optimize(List<VectorInterface> features, VectorInterface labels, VectorInterface weights);
}