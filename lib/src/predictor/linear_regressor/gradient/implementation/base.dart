import 'package:dart_ml/src/math/vector/vector.dart';
import 'package:dart_ml/src/predictor/interface/predictor.dart';
import 'package:dart_ml/src/optimizer/gradient/optimizer.dart';
import 'package:dart_ml/src/estimator/estimator.dart';
import 'package:dart_ml/src/estimator/estimator_type.dart';
import 'package:dart_ml/src/estimator/estimator_factory.dart';

abstract class GradientLinearRegressor implements Predictor {
  final Estimator estimator;
  final GradientOptimizer _optimizer;

  Vector _weights;

  GradientLinearRegressor(this._optimizer, {EstimatorType estimatorType})
      : estimator = EstimatorFactory.create(estimatorType);

  void train(List<Vector> features, Vector labels, {Vector weights}) {
    _weights = _optimizer.optimize(features, labels, weights: weights);
  }

  double test(List<Vector> features, Vector origLabels, {Estimator estimator}) {
    estimator = estimator ?? this.estimator;
    Vector prediction = predict(features);
    return estimator.calculateError(prediction, origLabels);
  }

  Vector predict(List<Vector> features) {
    List<double> labels = new List<double>(features.length);
    for (int i = 0; i < features.length; i++) {
      labels[i] = _weights.dot(features[i]);
    }
    return new Vector.from(labels);
  }
}
