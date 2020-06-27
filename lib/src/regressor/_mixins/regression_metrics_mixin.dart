import 'package:ml_algo/ml_algo.dart';

mixin RegressionMetricsMixin {
  List<MetricType> get allowedMetrics => [
    MetricType.mape,
    MetricType.rmse,
  ];
}
