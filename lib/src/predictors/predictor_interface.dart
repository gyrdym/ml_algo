import 'package:dart_ml/src/math/vector_interface.dart';
import 'package:dart_ml/src/optimizers/optimizer_interface.dart';

abstract class PredictorInterface<T extends VectorInterface> {
  void train(List<T> features, T labels);
  T predict(List<T> features);
  OptimizerInterface<T> get optimizer;
  T get weights;
}