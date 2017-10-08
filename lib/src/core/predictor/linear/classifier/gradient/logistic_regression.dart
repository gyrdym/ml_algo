part of 'package:dart_ml/src/core/implementation.dart';

class LogisticRegressor implements Classifier {
  _ClassifierBase _classifier;

  LogisticRegressor({
    int iterationLimit,

    double learningRate,
    double minWeightsDistance,
    double alpha,
    double argumentIncrement,

    ClassificationMetricType metric,
    Regularization regularization
  }) {
    coreInjector ??= new ModuleInjector([
      ModuleFactory.createLogisticRegressionModule(
        learningRate: learningRate,
        minWeightsDistance: minWeightsDistance,
        iterationLimit: iterationLimit,
        metricType: metric,
        regularization: regularization,
        alpha: alpha,
        argumentIncrement: argumentIncrement
      )
    ]);

    _classifier = new _ClassifierBase();
  }

  Metric get metric => _classifier.metric;

  void train(List<Float32x4Vector> features, List<double> origLabels, {Float32x4Vector weights}) =>
      _classifier.train(features, origLabels);

  double test(List<Float32x4Vector> features, List<double> origLabels, {MetricType metric}) =>
      _classifier.test(features, origLabels, metricType: metric);

  Float32x4Vector predict(List<Float32x4Vector> features) =>
      _classifier.predict(features);

  Float32x4Vector predictClasses(List<Float32x4Vector> features) =>
      _classifier.predictClasses(features);
}
