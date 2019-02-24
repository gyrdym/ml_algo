import 'package:ml_algo/src/metric/classification/accuracy.dart';
import 'package:ml_algo/src/metric/classification/type.dart';
import 'package:ml_algo/src/metric/metric.dart';

abstract class ClassificationMetricFactory {
  static Metric accuracy() => const AccuracyMetric();

  static Metric createByType(ClassificationMetricType type) {
    Metric metric;

    switch (type) {
      case ClassificationMetricType.accuracy:
        metric = accuracy();
        break;

      default:
        throw UnsupportedError(
            'Unsupported classification metric type: ${type}');
    }

    return metric;
  }
}
