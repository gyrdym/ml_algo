import 'dart:typed_data';

import 'package:dart_ml/src/metric/regression/metric.dart';
import 'package:linalg/linalg.dart';

class MAPEMetric implements RegressionMetric<Float32x4> {

  const MAPEMetric();

  @override
  double getError(Vector<Float32x4> predictedLabels, Vector<Float32x4> origLabels) =>
      100 / predictedLabels.length * ((origLabels - predictedLabels) / origLabels).abs().sum();
}
