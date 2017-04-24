import 'package:dart_ml/src/math/vector/vector_interface.dart';

abstract class Estimator {
  double calculateError(VectorInterface predictedLabels, VectorInterface origLabels);
}
