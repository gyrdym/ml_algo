part of 'package:dart_ml/src/core/implementation.dart';

class BGDRegressor implements Predictor {
  _PredictorBase _predictor;

  BGDRegressor({
    int iterationLimit,

    double learningRate,
    double minWeightsDistance,
    double alpha,
    double argumentIncrement,

    RegressionMetricType metric,
    Regularization regularization,
  }) {
    coreInjector ??= new ModuleInjector([
      ModuleFactory.createMBGDRegressionModule(
        learningRate: learningRate,
        minWeightsDistance: minWeightsDistance,
        iterationLimit: iterationLimit,
        metric: metric,
        regularization: regularization,
        alpha: alpha,
        argumentIncrement: argumentIncrement
      )
    ]);

    _predictor = new _PredictorBase();
  }

  Metric get metric => _predictor.metric;

  void train(List<Float32x4Vector> features, List<double> labels, {Float32x4Vector weights}) =>
      _predictor.train(features, labels, weights: weights);

  double test(List<Float32x4Vector> features, List<double> origLabels, {MetricType metric}) =>
      _predictor.test(features, origLabels, metricType: metric);

  Float32x4Vector predict(List<Float32x4Vector> features) =>
      _predictor.predict(features);
}
