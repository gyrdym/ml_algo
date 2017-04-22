import 'dart:math' as math;

import 'package:dart_ml/src/utils/generic_type_instantiator.dart';
import 'package:dart_ml/src/enums.dart';
import 'package:dart_ml/src/math/vector_interface.dart';
import 'package:dart_ml/src/optimizers/gradient_optimizer.dart';

class SGDOptimizer<T extends VectorInterface> extends GradientOptimizer<T> {
  final double minWeightsDistance;
  final double learningRate;
  final int iterationLimit;

  SGDOptimizer({this.learningRate = 1e-5, this.minWeightsDistance = 1e-8, this.iterationLimit = 10000});

  T optimize(List<T> features, T labels, CostFunction metric) {
    math.Random randomizer = new math.Random();
    double weightsDistance = double.MAX_FINITE;
    int iterationCounter = 0;

    T weights = Instantiator.createInstance(T, const Symbol('filled'), [features.first.length, 0.0]);

    while (weightsDistance > minWeightsDistance && iterationCounter < iterationLimit) {
      int k = randomizer.nextInt(features.length);
      double eta = learningRate / (iterationCounter + 1);
      VectorInterface newWeights = makeGradientStep(weights, [features[k]], [labels[k]], eta);
      weightsDistance = newWeights.distanceTo(weights);
      weights = newWeights;
      iterationCounter++;
    }

    return weights;
  }
}
