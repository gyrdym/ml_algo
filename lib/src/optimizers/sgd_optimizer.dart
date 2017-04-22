import 'dart:mirrors';
import 'dart:math' as math;

import 'package:dart_ml/src/utils/generic_type_instantiator.dart';
import 'package:dart_ml/src/enums.dart';
import 'package:dart_ml/src/math/vector_interface.dart';
import 'package:dart_ml/src/optimizers/optimizer_interface.dart';

class SGDOptimizer<T extends VectorInterface> extends OptimizerInterface<T> {
  final double minWeightsDistance;
  final double step;
  final int iterationLimit;

  SGDOptimizer({this.step = 1e-5, this.minWeightsDistance = 1e-8, this.iterationLimit = 10000});

  T optimize(List<T> features, T labels, CostFunction metric) {
    math.Random randomizer = new math.Random();
    double weightsDistance = double.MAX_FINITE;
    int iterationCounter = 0;

    T weights = Instantiator.createInstance(T, const Symbol('filled'), [features.first.length, 0.0]);

    while (weightsDistance > minWeightsDistance && iterationCounter < iterationLimit) {
      int k = randomizer.nextInt(features.length);
      double eta = step / (iterationCounter + 1);
      T newWeights = _doIteration(weights, features[k], labels[k], eta);
      weightsDistance = newWeights.distanceTo(weights);
      weights = newWeights;
      iterationCounter++;
    }

    return weights;
  }

  T _doIteration(T weights, T features, double y, double eta) {
    var newWeights = new List<double>.filled(features.length, 0.0);
    var delta = weights.vectorScalarMult(features) - y;

    for (int i = 0; i < features.length; i++) {
      newWeights[i] = weights[i] - (2 * eta) * delta * features[i];
    }

    return (reflectType(T) as ClassMirror).newInstance(const Symbol('from'), [newWeights]).reflectee;
  }
}
