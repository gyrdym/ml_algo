import 'package:dart_ml/src/math/vector_interface.dart';
import 'package:dart_ml/src/optimizers/optimizer_interface.dart';

abstract class GradientOptimizer<T extends VectorInterface> implements OptimizerInterface<T> {
  VectorInterface _makeGradientStep(VectorInterface Ks, VectorInterface Xs, double y, double eta) {
    var gradient = Xs.scalarMult(2 * eta * Ks.vectorScalarMult(Xs) - y);
    return Ks - gradient;
  }
}