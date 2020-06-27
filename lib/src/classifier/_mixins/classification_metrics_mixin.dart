import 'package:ml_algo/src/metric/metric_type.dart';

mixin ClassificationMetricsMixin {
  List<MetricType> get allowedMetrics => [
    MetricType.accuracy,
    MetricType.precision,
  ];
}
