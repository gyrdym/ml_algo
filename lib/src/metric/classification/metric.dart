import 'package:simd_vector/vector.dart';
import 'package:dart_ml/src/metric/base.dart';

part 'accuracy.dart';

abstract class ClassificationMetric implements Metric {
  double getError(Float32x4Vector predictedLabels, Float32x4Vector origLabels);
  double getScore(Float32x4Vector predictedLabels, Float32x4Vector origLabels);

  factory ClassificationMetric.Accuracy() => const _AccuracyMetric();
}
