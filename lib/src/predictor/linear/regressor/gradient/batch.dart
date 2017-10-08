part of 'package:dart_ml/src/predictor/implementation.dart';

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
    injector ??= new ModuleInjector([
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
  }

  Metric get metric => _predictor.metric;

  void train(List<Float32x4Vector> features, List<double> labels, {Float32x4Vector weights}) =>
      _predictor.train(features, labels, weights: weights);

  double test(List<Float32x4Vector> features, List<double> origLabels, {Metric metric}) =>
      _predictor.test(features, origLabels, metric: metric);

  Float32x4Vector predict(List<Float32x4Vector> features) =>
      _predictor.predict(features);
}
