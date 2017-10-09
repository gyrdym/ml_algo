part of 'package:dart_ml/src/core/implementation.dart';

class MetricFactory {
  static Metric createByType(MetricType type) {
    Metric metric;

    switch (type) {
      case MetricType.RMSE:
        metric = RegressionMetricFactory.RMSE();
        break;

      case MetricType.MAPE:
        metric = RegressionMetricFactory.MAPE();
        break;

      case MetricType.ACCURACY:
        metric = ClassificationMetricFactory.Accuracy();
        break;

      default:
        throw new UnsupportedError('Unsupported metric type $type');
    }

    return metric;
  }
}