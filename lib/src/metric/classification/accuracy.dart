import 'dart:typed_data';

import 'package:dart_ml/src/metric/classification/metric.dart';
import 'package:linalg/vector.dart';

class AccuracyMetric implements
    ClassificationMetric<Float32x4List, Float32List, Float32x4> {

  const AccuracyMetric();

  @override
  double getError(
    SIMDVector<Float32x4List, Float32List, Float32x4> predictedLabels,
    SIMDVector<Float32x4List, Float32List, Float32x4> origLabels
  ) => 1 - getScore(predictedLabels, origLabels);

  @override
  double getScore(
    SIMDVector<Float32x4List, Float32List, Float32x4> predictedLabels,
    SIMDVector<Float32x4List, Float32List, Float32x4> origLabels
  ) {
    double score = 0.0;
    for (int i = 0; i < origLabels.length; i++) {
      if (origLabels[i] == predictedLabels[i]) {
        score++;
      }
    }
    return score / origLabels.length;
  }
}
