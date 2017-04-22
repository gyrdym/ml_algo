import 'dart:math' as math;

import 'package:dart_ml/src/math/vector_interface.dart';
import 'package:dart_ml/src/optimizers/gradient_optimizer.dart';

class SGDOptimizer<T extends VectorInterface> extends GradientOptimizer<T> {
  final math.Random _randomizer = new math.Random();

  SGDOptimizer(double learningRate, double minWeightsDistance, int iterationLimit) : super(
      learningRate,
      minWeightsDistance,
      iterationLimit
    );

  @override
  T doIteration(T weights, List<T> features, List<double> labels, double eta) {
    int k = _randomizer.nextInt(features.length);
    return makeGradientStep(weights, [features[k]], [labels[k]], eta);
  }
}
