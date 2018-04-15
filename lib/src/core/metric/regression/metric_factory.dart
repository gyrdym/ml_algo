import 'package:dart_ml/src/core/metric/regression/mape.dart';
import 'package:dart_ml/src/core/metric/regression/metric.dart';
import 'package:dart_ml/src/core/metric/regression/rmse.dart';
import 'package:dart_ml/src/core/metric/regression/type.dart';

class RegressionMetricFactory {
  static RegressionMetric RMSE() => const RMSEMetric();
  static RegressionMetric MAPE() => const MAPEMetric();

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
