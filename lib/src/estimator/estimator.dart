import 'package:dart_vector/vector.dart';

abstract class Estimator {
  double calculateError(Float32x4Vector predictedLabels, Float32x4Vector origLabels);
}
