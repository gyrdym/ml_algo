import 'dart:math' as math;

import 'package:dart_ml/src/math/vector/vector.dart';
import 'package:dart_ml/src/optimizer/gradient/base_optimizer.dart';

class SGDOptimizer extends GradientOptimizer {
  final math.Random _random = new math.Random();

  SGDOptimizer(double learningRate, double minWeightsDistance, int iterationLimit) : super(
      learningRate,
      minWeightsDistance,
      iterationLimit
    );

  @override
  Vector iteration(Vector weights, List<Vector> features, Vector labels, double eta) {
    int k = _random.nextInt(features.length);
    return makeGradientStep(weights, [features[k]], new Vector.from([labels[k]]), eta);
  }
}
