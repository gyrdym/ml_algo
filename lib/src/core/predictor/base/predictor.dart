part of 'package:dart_ml/src/core/interface.dart';

abstract class Predictor {
  Metric get metric;
  void train(List<Float32x4Vector> features, List<double> labels, {Float32x4Vector weights});
  double test(List<Float32x4Vector> features, List<double> origLabels, {MetricType metric});
  Float32x4Vector predict(List<Float32x4Vector> features);
}
