import 'dart:math' as math;

import 'package:dart_ml/src/optimizers/optimizer.dart';
import 'package:dart_ml/src/vector_operations.dart' as vectors;

class StochasticGradientDescent implements Optimizer {
  double step;
  double minWeightsDistance;
  int iterationLimit;
  List<double> errors;

  StochasticGradientDescent({this.step = 1e-9, this.minWeightsDistance = 1e-8, this.iterationLimit = 1000});

  List<double> optimize(List<List<double>> features, List<double> labels) {
    math.Random randomizer = new math.Random();
    double weightsDistance = double.MAX_FINITE;
    int iterationCounter = 0;

    List<double> weights = vectors.create(features.first.length, 0.0);

    while (weightsDistance > minWeightsDistance && iterationCounter < iterationLimit) {
      int k = randomizer.nextInt(features.length);

      double eta = step / (iterationCounter + 1);
      List<double> newWeights = _doIteration(weights, features[k], labels[k], eta);
      weightsDistance = vectors.distance(newWeights, weights);
      weights = newWeights;

      iterationCounter++;
    }

    return weights;
  }

  List<double> _doIteration(List<double> weights, List<double> features, double y, double eta) {
    int dimensions = features.length;
    List<double> newWeights = new List<double>();
    double delta = vectors.scalarMult(weights, features) - y;

    for (int i = 0; i < dimensions; i++) {
      double w = weights[i];
      double x = features[i];

      newWeights.add(w - (2 * eta / dimensions) * delta * x);
    }

    return newWeights;
  }
}