import 'package:ml_algo/src/metric/metric_type.dart';

class InvalidMetricTypeException implements Exception {
  InvalidMetricTypeException(MetricType metricType,
      List<MetricType> allowedTypes) :
        message = 'Inappropriate metric provided, allowed metrics: '
            '$allowedTypes, $metricType given';

  final String message;

  @override
  String toString() => message;
}
