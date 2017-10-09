part of 'package:dart_ml/src/core/implementation.dart';

class _ClassifierImpl implements Classifier {
  _ClassifierBase _classifier;

  _ClassifierImpl(Module module) {
    coreInjector ??= new ModuleInjector([module]);
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