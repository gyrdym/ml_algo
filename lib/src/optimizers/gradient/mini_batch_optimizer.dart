import 'dart:math' as math;
import 'package:dart_ml/src/math/vector/vector_interface.dart';
import 'package:dart_ml/src/optimizers/gradient/base_optimizer.dart';

class MBGDOptimizer extends GradientOptimizer {
  final math.Random _randomizer = new math.Random();

  MBGDOptimizer(double learningRate, double minWeightsDistance, int iterationLimit) : super(
      learningRate,
      minWeightsDistance,
      iterationLimit
    );

  @override
  VectorInterface iteration(VectorInterface weights, List<VectorInterface> features, VectorInterface labels, double eta) {
    int end = _randomizer.nextInt(features.length - 1) + 1;
    int start = _randomizer.nextInt(end);

    List<VectorInterface> featuresBatch = features.sublist(start, end);
    VectorInterface labelsBatch = labels.cut(start, end);

    return makeGradientStep(weights, featuresBatch, labelsBatch, eta);
  }
}
