import 'package:dart_vector/vector.dart';
import 'package:dart_ml/src/predictor/interface/predictor.dart';
import 'package:dart_ml/src/optimizer/gradient/interface/base.dart';
import 'package:dart_ml/src/estimator/estimator.dart';
import 'package:dart_ml/src/estimator/estimator_type.dart';
import 'package:dart_ml/src/estimator/estimator_factory.dart';

abstract class GradientLinearRegressor implements Predictor {
  final Estimator estimator;
  final GradientOptimizer _optimizer;

  Float32x4Vector _weights;

  GradientLinearRegressor(this._optimizer, {EstimatorType estimatorType})
      : estimator = EstimatorFactory.create(estimatorType);

  void train(List<Float32x4Vector> features, List<double> labels, {Float32x4Vector weights}) {
    _weights = _optimizer.optimize(features, labels, weights: weights);
  }

  double test(List<Float32x4Vector> features, List<double> origLabels, {Estimator estimator}) {
    estimator = estimator ?? this.estimator;
    Float32x4Vector prediction = predict(features);
    return estimator.calculateError(prediction, new Float32x4Vector.from(origLabels));
  }

  Float32x4Vector predict(List<Float32x4Vector> features) {
    List<double> labels = new List<double>(features.length);
    for (int i = 0; i < features.length; i++) {
      labels[i] = _weights.dot(features[i]);
    }
    return new Float32x4Vector.from(labels);
  }
}
