import 'dart:typed_data';

import 'package:ml_algo/src/metric/regression/metric.dart';
import 'package:ml_linalg/linalg.dart';

class MAPEMetric implements RegressionMetric<Float32x4> {
  const MAPEMetric();

  @override
  double getError(MLVector<Float32x4> predictedLabels, MLVector<Float32x4> origLabels) =>
      100 / predictedLabels.length * ((origLabels - predictedLabels) / origLabels).abs().sum();
}
