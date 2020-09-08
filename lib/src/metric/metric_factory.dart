import 'package:ml_algo/src/metric/metric.dart';
import 'package:ml_algo/src/metric/metric_type.dart';

abstract class MetricFactory {
  Metric createByType(MetricType metricType);
}
