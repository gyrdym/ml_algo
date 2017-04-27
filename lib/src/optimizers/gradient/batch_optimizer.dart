import 'package:dart_ml/src/math/vector/vector_interface.dart';
import 'package:dart_ml/src/optimizers/gradient/base_optimizer.dart';

class BGDOptimizer extends GradientOptimizer {
  BGDOptimizer(double learningRate, double minWeightsDistance, int iterationLimit) : super(
      learningRate,
      minWeightsDistance,
      iterationLimit
    );

  @override
  VectorInterface iteration(VectorInterface weights, List<VectorInterface> features, List<double> labels, double eta) {
    return makeGradientStep(weights, features, labels, eta);
  }
}
