import 'package:dart_ml/src/metric/classification/accuracy.dart';
import 'package:dart_ml/src/metric/classification/metric.dart';
import 'package:dart_ml/src/metric/classification/type.dart';

abstract class ClassificationMetricFactory {
  static ClassificationMetric Accuracy() => const AccuracyMetric();

  static ClassificationMetric createByType(ClassificationMetricType type) {
    ClassificationMetric metric;

    switch (type) {
      case ClassificationMetricType.ACCURACY:
        metric = Accuracy();
        break;

      default:
        throw new UnsupportedError('Unsupported classification metric type: ${type}');
    }

    return metric;
  }
}
