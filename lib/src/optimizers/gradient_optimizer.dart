import 'package:dart_ml/src/utils/generic_type_instantiator.dart';
import 'package:dart_ml/src/math/vector_interface.dart';
import 'package:dart_ml/src/optimizers/optimizer_interface.dart';

abstract class GradientOptimizer<T extends VectorInterface> implements OptimizerInterface<T> {
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