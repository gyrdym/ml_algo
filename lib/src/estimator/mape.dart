import 'package:dart_ml/src/estimator/estimator.dart';
import 'package:simd_vector/vector.dart';

class MAPEEstimator implements Estimator {
  double calculateError(Float32x4Vector predictedLabels, Float32x4Vector origLabels) =>
      100 / predictedLabels.length * ((origLabels - predictedLabels) / origLabels).abs().sum();
}
