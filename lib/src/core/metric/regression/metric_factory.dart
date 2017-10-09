part of 'package:dart_ml/src/core/implementation.dart';

class RegressionMetricFactory {
  static RegressionMetric RMSE() => const _RMSEMetric();
  static RegressionMetric MAPE() => const _MAPEMetric();

  static RegressionMetric createByType(RegressionMetricType type) {
    RegressionMetric metric;

    switch (type) {
      case RegressionMetricType.MAPE:
        metric = MAPE();
        break;

      case RegressionMetricType.RMSE:
        metric = RMSE();
        break;

      default:
        throw new UnsupportedError('Unsupported regression metric type: ${type}');
    }

    return metric;
  }
}
