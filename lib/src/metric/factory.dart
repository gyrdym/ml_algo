import 'package:dart_ml/src/metric/classification/metric_factory.dart';
import 'package:dart_ml/src/metric/metric.dart';
import 'package:dart_ml/src/metric/regression/metric_factory.dart';
import 'package:dart_ml/src/metric/type.dart';

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