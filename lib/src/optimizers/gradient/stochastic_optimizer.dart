import 'dart:math' as math;

import 'package:dart_ml/src/math/vector/vector_interface.dart';
import 'package:dart_ml/src/optimizers/gradient/base_optimizer.dart';

class SGDOptimizer extends GradientOptimizer {
  final math.Random _randomizer = new math.Random();

  SGDOptimizer(double learningRate, double minWeightsDistance, int iterationLimit) : super(
      learningRate,
      minWeightsDistance,
      iterationLimit
    );

  @override
  VectorInterface iteration(VectorInterface weights, List<VectorInterface> features, VectorInterface labels, double eta) {
    int k = _randomizer.nextInt(features.length);
    return makeGradientStep(weights, [features[k]], labels.createFrom([labels[k]]), eta);
  }
}
