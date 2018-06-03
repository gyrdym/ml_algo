import 'package:dart_ml/src/metric/type.dart';
import 'package:linalg/vector.dart';

abstract class Evaluable {
  void fit(Iterable<Vector> features, Vector origLabels, {Vector initialWeights, bool isDataNormalized});
  double test(Iterable<Vector> features, Vector origLabels, MetricType metric);
}
