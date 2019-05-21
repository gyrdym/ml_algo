import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_linalg/matrix.dart';

abstract class Assessable {
  /// Assesses model according to provided [metric]
  double assess(Matrix observations, Matrix outcomes, MetricType metric);
}
