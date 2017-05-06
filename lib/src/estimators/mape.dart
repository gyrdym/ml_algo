import 'package:dart_ml/src/estimators/estimator_interface.dart';
import 'package:dart_ml/src/math/vector/vector_interface.dart';

class MAPEEstimator implements EstimatorInterface {
  double calculateError(VectorInterface predictedLabels, VectorInterface origLabels) =>
      100 / predictedLabels.length * ((origLabels - predictedLabels) / origLabels).abs(inPlace: true).sum();
}
