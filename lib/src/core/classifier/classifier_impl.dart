import 'package:dart_ml/src/core/classifier/classifier.dart';
import 'package:dart_ml/src/core/classifier/classifier_base.dart';
import 'package:dart_ml/src/core/metric/metric.dart';
import 'package:dart_ml/src/core/metric/type.dart';
import 'package:dart_ml/src/di/injector.dart';
import 'package:di/di.dart';
import 'package:simd_vector/vector.dart';

class ClassifierImpl implements Classifier {
  ClassifierBase _classifier;

  ClassifierImpl(Module module) {
    coreInjector ??= new ModuleInjector([module]);
    _classifier = new ClassifierBase();
  }

  @override
  Metric get metric => _classifier.metric;

  @override
  void train(
    List<Float32x4Vector> features,
    List<double> origLabels,
    {Float32x4Vector weights}
  ) => _classifier.train(features, origLabels);

  @override
  double test(
    List<Float32x4Vector> features,
    List<double> origLabels,
    {MetricType metric}
  ) => _classifier.test(features, origLabels, metric: metric);

  @override
  Float32x4Vector predict(List<Float32x4Vector> features) =>
      _classifier.predict(features);

  @override
  Float32x4Vector predictClasses(List<Float32x4Vector> features) =>
      _classifier.predictClasses(features);
}