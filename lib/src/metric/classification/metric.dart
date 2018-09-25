import 'package:dart_ml/src/metric/metric.dart';
import 'package:linalg/vector.dart';

abstract class ClassificationMetric implements Metric{
  @override
  double getError(Vector predictedLabels, Vector origLabels);
  double getScore(Vector predictedLabels, Vector origLabels);
}