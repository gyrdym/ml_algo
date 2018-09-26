import 'package:dart_ml/src/metric/regression/mape.dart';
import 'package:dart_ml/src/metric/regression/metric.dart';
import 'package:dart_ml/src/metric/regression/rmse.dart';
import 'package:dart_ml/src/metric/regression/type.dart';

class RegressionMetricFactory {
  static RegressionMetric rmse() => const RMSEMetric();
  static RegressionMetric mape() => const MAPEMetric();

  static RegressionMetric createByType(RegressionMetricType type) {
    RegressionMetric metric;

    switch (type) {
      case RegressionMetricType.mape:
        metric = mape();
        break;

      case RegressionMetricType.rmse:
        metric = rmse();
        break;

      default:
        throw UnsupportedError('Unsupported regression metric type: ${type}');
    }

    return metric;
  }
}
