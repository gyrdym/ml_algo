import 'package:dart_ml/src/estimators/estimator.dart';
import 'package:dart_ml/src/math/vector_interface.dart';

class MAPEEstimator implements Estimator {
  double calculateError(VectorInterface predictedLabels, VectorInterface origLabels) =>
      100 / predictedLabels.length * ((origLabels - predictedLabels) / origLabels).abs(inPlace: true).sum();
}