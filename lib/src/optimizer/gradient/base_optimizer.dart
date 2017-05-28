import 'package:dart_ml/src/math/vector/vector.dart';
import 'package:dart_ml/src/optimizer/optimizer.dart';

abstract class GradientOptimizer implements Optimizer {
  final double minWeightsDistance;
  final double learningRate;
  final int iterationLimit;

  GradientOptimizer(this.learningRate, this.minWeightsDistance, this.iterationLimit);

  Vector optimize(List<Vector> features, Vector labels, {Vector weights}) {
    weights = weights ?? new Vector.zero(features.first.length);
    double weightsDistance = double.MAX_FINITE;
    int iterationCounter = 0;

    while (weightsDistance > minWeightsDistance && iterationCounter < iterationLimit) {
      double eta = learningRate / ++iterationCounter;
      Vector newWeights = iteration(weights, features, labels, eta);
      weightsDistance = newWeights.distanceTo(weights);
      weights = newWeights;
    }

    return weights;
  }

  Vector iteration(Vector weights, List<Vector> features, Vector labels, double eta);

  Vector makeGradientStep(Vector k, List<Vector> Xs, Vector y, double eta) {
    Vector gradientSumVector = _calculateGradient(k, Xs[0], y[0]);

    for (int i = 1; i < Xs.length; i++) {
      gradientSumVector += _calculateGradient(k, Xs[i], y[i]);
    }

    return k - gradientSumVector.scalarDiv(Xs.length * 1.0).scalarMul(2 * eta);
  }

  Vector _calculateGradient(Vector k, Vector x, double y) =>
      x.scalarMul((x.dot(k) - y));
}
