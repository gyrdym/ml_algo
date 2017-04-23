import 'dart:math' as math;
import 'package:dart_ml/src/math/vector_interface.dart';
import 'package:dart_ml/src/optimizers/gradient/base_gradient_optimizer.dart';

class MBGDOptimizer<T extends VectorInterface> extends GradientOptimizer<T> {
  math.Random _randomizer = new math.Random();

  MBGDOptimizer(double learningRate, double minWeightsDistance, int iterationLimit) : super(
      learningRate,
      minWeightsDistance,
      iterationLimit
    );

  @override
  T doIteration(T weights, List<T> features, List<double> labels, double eta) {
    _randomizer = new math.Random(_randomizer.nextInt(features.length));

    int end = _randomizer.nextInt(features.length);
    int start = _randomizer.nextInt(end);

    List<T> featuresBatch = features.sublist(start, end);
    List<double> labelsBatch = labels.sublist(start, end);

    return makeGradientStep(weights, featuresBatch, labelsBatch, eta);
  }
}
