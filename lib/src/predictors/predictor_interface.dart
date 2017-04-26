import 'package:dart_ml/src/math/vector/vector_interface.dart';
import 'package:dart_ml/src/optimizers/gradient/base_optimizer.dart';

abstract class PredictorInterface<T extends VectorInterface> {
  GradientOptimizer<T> optimizer;
  void train(List<T> features, List<double> labels);
  T predict(List<T> features);
  T get weights;
}
