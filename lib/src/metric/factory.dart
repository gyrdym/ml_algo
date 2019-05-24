import 'package:ml_algo/src/metric/classification/accuracy.dart';
import 'package:ml_algo/src/metric/metric.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/metric/regression/mape.dart';
import 'package:ml_algo/src/metric/regression/rmse.dart';

class MetricFactory {
  static Metric createByType(MetricType type) {
    switch (type) {
      case MetricType.rmse:
        return const RMSEMetric();

      case MetricType.mape:
        return const MAPEMetric();

      case MetricType.accuracy:
        return const AccuracyMetric();

      default:
        throw UnsupportedError('Unsupported metric type $type');
    }
  }
}
