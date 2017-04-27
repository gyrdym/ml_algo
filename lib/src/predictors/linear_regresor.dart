import 'package:dart_ml/src/math/vector/vector_interface.dart';
import 'package:dart_ml/src/predictors/predictor_interface.dart';
import 'package:dart_ml/src/optimizers/gradient/base_optimizer.dart';

abstract class LinearRegressor implements PredictorInterface {
  GradientOptimizer optimizer;
  VectorInterface _weights;

  VectorInterface get weights => _weights;

  void train(List<VectorInterface> features, List<double> labels, VectorInterface weights) {
    _weights = optimizer.optimize(features, labels, weights);
  }

  VectorInterface predict(List<VectorInterface> features, VectorInterface labels) {
    for (int i = 0; i < features.length; i++) {
      labels[i] = _weights.vectorScalarMult(features[i]);
    }

    return labels;
  }
}
