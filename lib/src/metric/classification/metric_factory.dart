part of 'package:dart_ml/src/implementation.dart';

abstract class ClassificationMetricFactory {
  static ClassificationMetric Accuracy() => const _AccuracyMetric();
}
