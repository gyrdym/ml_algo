import 'package:dart_ml/src/utils/generic_type_instantiator.dart';
import 'package:dart_ml/src/math/vector/vector_interface.dart';
import 'package:dart_ml/src/optimizers/optimizer_interface.dart';

abstract class GradientOptimizer<T extends VectorInterface> implements OptimizerInterface<T> {
  final double minWeightsDistance;
  final double learningRate;
  final int iterationLimit;

  GradientOptimizer(this.learningRate, this.minWeightsDistance, this.iterationLimit);

  T optimize(List<T> features, List<double> labels) {
    double weightsDistance = double.MAX_FINITE;
    int iterationCounter = 0;

    T weights = Instantiator.createInstance(T, const Symbol('filled'), [features.first.length, 0.0]);

    while (weightsDistance > minWeightsDistance && iterationCounter < iterationLimit) {
      double eta = learningRate / (iterationCounter + 1);
      T newWeights = doIteration(weights, features, labels, eta);
      weightsDistance = newWeights.distanceTo(weights);
      weights = newWeights;
      iterationCounter++;
    }

    return weights;
  }

  T doIteration(T weights, List<T> features, List<double> labels, double eta);

  T makeGradientStep(T k, List<T> Xs, List<double> y, double eta) {
    T gradientSumVector = Instantiator.createInstance(T, new Symbol('filled'), [k.length, 0.0]);

    for (int i = 0; i < Xs.length; i++) {
      gradientSumVector += _calculateGradient(k, Xs[i], y[i]);
    }

    return k - gradientSumVector.scalarDivision(Xs.length * 1.0).scalarMult(2 * eta);
  }

  T _calculateGradient(T k, T x, double y) =>
      x.scalarMult(x.vectorScalarMult(k) - y);
}