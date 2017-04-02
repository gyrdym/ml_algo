import 'dart:math' as math;

import 'package:dart_ml/src/optimizers/optimizer.dart';
import 'package:dart_ml/src/vector_operations.dart' as vectors;

class StochasticGradientDescent implements Optimizer {
  double eta;
  double minWeightsDistance;
  int iterationLimit;
  List<double> errors;

  StochasticGradientDescent({this.eta = 1e-2, this.minWeightsDistance = 1e-8, this.iterationLimit = 1000});

  List<double> optimize(List<List<num>> features, List<num> labels) {
    math.Random randomizer = new math.Random();
    double weightsDistance = double.MAX_FINITE;
    int iterationCounter = 0;

    List<double> weights = vectors.create(features.first.length, 0.0);

    while (weightsDistance > minWeightsDistance && iterationCounter < iterationLimit) {
      int k = randomizer.nextInt(features.length - 1);

      List<double> newWeights = _doStep(weights, features[k], labels[k], eta);
      weightsDistance = vectors.distance(newWeights, weights);
      weights = newWeights;

      iterationCounter++;
    }

    return weights;
  }

  List<double> _doStep(List<num> weights, List<num> features, num y, double eta) {
    int dimensions = features.length;
    List<double> newWeights = new List<double>();
    double diff = vectors.scalarMult(weights, features) - y;

    for (int i = 0; i < dimensions; i++) {
      double w = weights[i];
      num x = features[i];

      newWeights.add(w - (2 * eta / dimensions) * x * diff);
    }

    return newWeights;
  }
}