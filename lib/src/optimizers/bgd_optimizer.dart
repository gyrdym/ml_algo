import 'package:dart_ml/src/math/vector_interface.dart';
import 'package:dart_ml/src/optimizers/gradient_optimizer.dart';

class BGDOptimizer<T extends VectorInterface> extends GradientOptimizer<T> {
  BGDOptimizer(double learningRate, double minWeightsDistance, int iterationLimit) : super(
      learningRate,
      minWeightsDistance,
      iterationLimit
    );

  @override
  T doIteration(T weights, List<T> features, List<double> labels, double eta) {
    return makeGradientStep(weights, features, labels, eta);
  }
}
