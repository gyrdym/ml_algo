import 'package:dart_ml/src/math/vector/vector_interface.dart';
import 'package:dart_ml/src/optimizers/gradient/base_optimizer.dart';

abstract class PredictorInterface {
  GradientOptimizer optimizer;
  void train(List<VectorInterface> features, List<double> labels, VectorInterface weights);
  VectorInterface predict(List<VectorInterface> features, VectorInterface weights);
  VectorInterface get weights;
}
