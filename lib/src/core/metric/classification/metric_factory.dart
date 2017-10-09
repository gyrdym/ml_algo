part of 'package:dart_ml/src/core/implementation.dart';

abstract class ClassificationMetricFactory {
  static ClassificationMetric Accuracy() => const _AccuracyMetric();

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
