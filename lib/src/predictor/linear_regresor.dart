import 'package:dart_ml/src/math/vector/vector.dart';
import 'package:dart_ml/src/predictor/predictor.dart';
import 'package:dart_ml/src/optimizer/optimizer.dart';
import 'package:dart_ml/src/estimator/estimator.dart';

abstract class LinearRegressor implements Predictor {
  Estimator defaultEstimator;
  Optimizer optimizer;
  Vector _weights;

  void train(List<Vector> features, Vector labels, Vector weights) {
    _weights = optimizer.optimize(features, labels, weights);
  }

  double test(List<Vector> features, Vector origLabels, {Estimator estimator}) {
    estimator = estimator ?? defaultEstimator;
    Vector prediction = predict(features, origLabels.copy()..fill(0.0));
    return estimator.calculateError(prediction, origLabels);
  }

  Vector predict(List<Vector> features, Vector labels) {
    for (int i = 0; i < features.length; i++) {
      labels[i] = _weights.dot(features[i]);
    }

    return labels;
  }
}
