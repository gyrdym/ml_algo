import 'package:dart_ml/src/metric/type.dart';
import 'package:simd_vector/vector.dart';

abstract class Evaluable {
  void fit(Iterable<Vector> features, Iterable<double> origLabels, {Vector initialWeights, bool isDataNormalized});
  double test(Iterable<Vector> features, Iterable<double> origLabels, MetricType metric);
}
