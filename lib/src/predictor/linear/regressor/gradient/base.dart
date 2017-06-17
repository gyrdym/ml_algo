import 'package:dart_ml/src/math/vector/vector.dart';
import 'package:dart_ml/src/predictor/interface/predictor.dart';
import 'package:dart_ml/src/optimizer/gradient/interface/base.dart';
import 'package:dart_ml/src/metric/metric.dart';

abstract class GradientLinearRegressor implements Predictor {
  final Metric metric;
  final GradientOptimizer _optimizer;

  Vector _weights;

  GradientLinearRegressor(this._optimizer, {Metric metric}) :
        metric = metric ?? new Metric.RMSE();

  void train(List<Vector> features, Vector labels, {Vector weights}) {
    _weights = _optimizer.optimize(features, labels, weights: weights);
  }

  double test(List<Vector> features, Vector origLabels, {Metric estimator}) {
    estimator = estimator ?? this.metric;
    Vector prediction = predict(features);
    return estimator.getError(prediction, origLabels);
  }

  Vector predict(List<Vector> features) {
    List<double> labels = new List<double>(features.length);
    for (int i = 0; i < features.length; i++) {
      labels[i] = _weights.dot(features[i]);
    }
    return new Vector.from(labels);
  }
}
