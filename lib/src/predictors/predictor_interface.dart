import 'package:dart_ml/src/math/vector/vector_interface.dart';
import 'package:dart_ml/src/estimators/estimator_interface.dart';

abstract class PredictorInterface {
  void train(List<VectorInterface> features, VectorInterface labels, VectorInterface weights);
  double test(List<VectorInterface> features, VectorInterface origLabels, {EstimatorInterface estimator});
  VectorInterface predict(List<VectorInterface> features, VectorInterface labels);
}
