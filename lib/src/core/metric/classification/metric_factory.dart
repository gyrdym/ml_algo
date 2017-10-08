part of 'package:dart_ml/src/core/implementation.dart';

abstract class ClassificationMetricFactory {
  static ClassificationMetric Accuracy() => const _AccuracyMetric();
}
