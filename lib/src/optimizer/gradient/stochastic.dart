import 'dart:math' as math;

import 'package:dart_ml/src/optimizer/gradient/optimizer.dart';
import 'package:dart_ml/src/optimizer/regularization.dart';

class SGDOptimizer extends GradientOptimizer {
  final math.Random _random = new math.Random();

  SGDOptimizer(double learningRate, double minWeightsDistance, int iterationLimit, Regularization regularization)
      : super(learningRate, minWeightsDistance, iterationLimit, regularization);

  @override
  Iterable<int> getSampleRange(int totalSamplesCount) {
    int k = _random.nextInt(totalSamplesCount);
    return [k, k + 1];
  }
}
