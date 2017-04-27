import 'package:dart_ml/src/math/vector/vector_interface.dart';
import 'package:dart_ml/src/optimizers/optimizer_interface.dart';

abstract class GradientOptimizer implements OptimizerInterface {
  final double minWeightsDistance;
  final double learningRate;
  final int iterationLimit;

  GradientOptimizer(this.learningRate, this.minWeightsDistance, this.iterationLimit);

  VectorInterface optimize(List<VectorInterface> features, List<double> labels, VectorInterface weights) {
    double weightsDistance = double.MAX_FINITE;
    int iterationCounter = 0;

    while (weightsDistance > minWeightsDistance && iterationCounter < iterationLimit) {
      double eta = learningRate / (iterationCounter + 1);
      VectorInterface newWeights = iteration(weights, features, labels, eta);
      weightsDistance = newWeights.distanceTo(weights);
      weights = newWeights;
      iterationCounter++;
    }

    return weights;
  }

  VectorInterface iteration(VectorInterface weights, List<VectorInterface> features, List<double> labels, double eta);

  VectorInterface makeGradientStep(VectorInterface k, List<VectorInterface> Xs, List<double> y, double eta) {
    VectorInterface gradientSumVector = _calculateGradient(k, Xs[0], y[0]);

    for (int i = 1; i < Xs.length; i++) {
      gradientSumVector += _calculateGradient(k, Xs[i], y[i]);
    }

    return k - gradientSumVector.scalarDivision(Xs.length * 1.0).scalarMult(2 * eta);
  }

  VectorInterface _calculateGradient(VectorInterface k, VectorInterface x, double y) =>
      x.scalarMult(x.vectorScalarMult(k) - y);
}
