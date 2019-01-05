import 'package:ml_algo/src/metric/classification/metric_factory.dart';
import 'package:ml_algo/src/metric/metric.dart';
import 'package:ml_algo/src/metric/regression/metric_factory.dart';
import 'package:ml_algo/metric_type.dart';

class MetricFactory {
  static Metric createByType(MetricType type) {
    Metric metric;

    switch (type) {
      case MetricType.rmse:
        metric = RegressionMetricFactory.rmse();
        break;

      case MetricType.mape:
        metric = RegressionMetricFactory.mape();
        break;

      case MetricType.accuracy:
        metric = ClassificationMetricFactory.accuracy();
        break;

      default:
        throw UnsupportedError('Unsupported metric type $type');
    }

    return metric;
  }
}
