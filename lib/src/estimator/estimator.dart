import 'package:dart_ml/src/math/vector/vector.dart';

abstract class Estimator {
  double calculateError(Vector predictedLabels, Vector origLabels);
}
