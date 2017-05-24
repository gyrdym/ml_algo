import 'dart:math' as math;
import 'package:dart_ml/src/math/vector/vector.dart';
import 'package:dart_ml/src/optimizer/gradient/base_optimizer.dart';

class MBGDOptimizer extends GradientOptimizer {
  final math.Random _randomizer = new math.Random();

  MBGDOptimizer(double learningRate, double minWeightsDistance, int iterationLimit) : super(
      learningRate,
      minWeightsDistance,
      iterationLimit
    );

  @override
  Vector iteration(Vector weights, List<Vector> features, Vector labels, double eta) {
    int end = _randomizer.nextInt(features.length - 1) + 1;
    int start = _randomizer.nextInt(end);

    List<Vector> featuresBatch = features.sublist(start, end);
    Vector labelsBatch = labels.cut(start, end);

    return makeGradientStep(weights, featuresBatch, labelsBatch, eta);
  }
}
