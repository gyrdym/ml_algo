import 'dart:math' as math;

import 'package:dart_ml/src/optimizer/gradient/base_optimizer.dart';

class SGDOptimizer extends GradientOptimizer {
  final math.Random _random = new math.Random();

  SGDOptimizer(double learningRate, double minWeightsDistance, int iterationLimit) : super(
      learningRate,
      minWeightsDistance,
      iterationLimit
    );

  @override
  Iterable<int> getSampleRange(int totalSamplesCount) {
    int k = _random.nextInt(totalSamplesCount);
    return [k, k + 1];
  }
}
