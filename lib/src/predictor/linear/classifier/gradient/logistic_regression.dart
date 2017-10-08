part of 'package:dart_ml/src/predictor/implementation.dart';

class LogisticRegressor implements Classifier {
  _ClassifierBase _classifier;

  LogisticRegressor({
    double learningRate,
    double minWeightsDistance,
    int iterationLimit,
    ClassificationMetricType metric,
    Regularization regularization,
    double alpha,
    double argumentIncrement
  }) {
    injector ??= new ModuleInjector([
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

  double test(List<Float32x4Vector> features, List<double> origLabels, {Metric metric}) =>
      _classifier.test(features, origLabels, metric: metric);

  Float32x4Vector predict(List<Float32x4Vector> features) =>
      _classifier.predict(features);

  Float32x4Vector predictClasses(List<Float32x4Vector> features) =>
      _classifier.predictClasses(features);
}
