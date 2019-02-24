import 'package:ml_algo/src/metric/metric.dart';
import 'package:ml_algo/src/metric/regression/mape.dart';
import 'package:ml_algo/src/metric/regression/rmse.dart';
import 'package:ml_algo/src/metric/regression/type.dart';

class RegressionMetricFactory {
  static Metric rmse() => const RMSEMetric();

  static Metric mape() => const MAPEMetric();

  static Metric createByType(RegressionMetricType type) {
    Metric metric;

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
