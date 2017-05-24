import 'package:dart_ml/src/math/vector/vector.dart';
import 'package:dart_ml/src/optimizer/gradient/base_optimizer.dart';

class BGDOptimizer extends GradientOptimizer {
  BGDOptimizer(double learningRate, double minWeightsDistance, int iterationLimit) : super(
      learningRate,
      minWeightsDistance,
      iterationLimit
    );

  @override
  Vector iteration(Vector weights, List<Vector> features, Vector labels, double eta) {
    return makeGradientStep(weights, features, labels, eta);
  }
}
