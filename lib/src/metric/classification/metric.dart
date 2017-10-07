part of 'package:dart_ml/src/interface.dart';

abstract class ClassificationMetric {
  double getError(Float32x4Vector predictedLabels, Float32x4Vector origLabels);
  double getScore(Float32x4Vector predictedLabels, Float32x4Vector origLabels);
}