import 'package:dart_ml/src/math/vector/vector.dart';
import 'package:dart_ml/src/metric/metric.dart';

abstract class Predictor {
  Metric get metric;
  void train(List<Vector> features, Vector labels, {Vector weights});
  double test(List<Vector> features, Vector origLabels, {Metric estimator});
  Vector predict(List<Vector> features);
}
