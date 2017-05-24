import 'package:dart_ml/src/estimator/estimator.dart';
import 'package:dart_ml/src/math/vector/vector.dart';

class MAPEEstimator implements Estimator {
  double calculateError(Vector predictedLabels, Vector origLabels) =>
      100 / predictedLabels.length * ((origLabels - predictedLabels) / origLabels).abs(inPlace: true).sum();
}
