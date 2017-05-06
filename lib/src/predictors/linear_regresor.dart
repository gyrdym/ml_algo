import 'package:dart_ml/src/math/vector/vector_interface.dart';
import 'package:dart_ml/src/predictors/predictor_interface.dart';
import 'package:dart_ml/src/optimizers/optimizer_interface.dart';
import 'package:dart_ml/src/estimators/estimator_interface.dart';

abstract class LinearRegressor implements PredictorInterface {
  OptimizerInterface optimizer;
  VectorInterface _weights;

  void train(List<VectorInterface> features, VectorInterface labels, VectorInterface weights) {
    _weights = optimizer.optimize(features, labels, weights);
  }

  double test(List<VectorInterface> features, VectorInterface origLabels, {EstimatorInterface estimator}) {
    VectorInterface prediction = predict(features, origLabels.copy()..fill(0.0));
    return estimator.calculateError(prediction, origLabels);
  }

  VectorInterface predict(List<VectorInterface> features, VectorInterface labels) {
    for (int i = 0; i < features.length; i++) {
      labels[i] = _weights.vectorScalarMult(features[i]);
    }

    return labels;
  }
}
