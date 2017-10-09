part of 'package:dart_ml/src/core/implementation.dart';

class _RegressorImpl implements Predictor {
  _PredictorBase _predictor;

  _RegressorImpl(Module module) {
    coreInjector ??= new ModuleInjector([module]);
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
