import 'package:ml_algo/src/metric/metric.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

abstract class QualityEvaluator {
  Metric get metric;
  double evaluate(MLMatrix features, MLVector origLabels,
      MetricType metricType);
}
