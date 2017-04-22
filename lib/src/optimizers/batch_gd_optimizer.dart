import 'package:dart_ml/src/math/vector_interface.dart';
import 'package:dart_ml/src/optimizers/gradient_optimizer.dart';

class BatchGDOptimizer<T extends VectorInterface> extends GradientOptimizer<T> {
  BatchGDOptimizer(double learningRate, double minWeightsDistance, int iterationLimit) : super(
      learningRate,
      minWeightsDistance,
      iterationLimit
    );

  @override
  T doIteration(T weights, List<T> features, T labels, double eta) {
    return makeGradientStep(weights, features, labels as List, eta);
  }
}
