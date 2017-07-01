import 'package:dart_vector/vector.dart' show Vector;

abstract class Estimator {
  double calculateError(Vector predictedLabels, Vector origLabels);
}
