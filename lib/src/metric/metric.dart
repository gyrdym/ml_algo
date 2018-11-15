import 'package:linalg/vector.dart';

abstract class Metric<S extends List<E>, T extends List<double>, E> {
  double getError(SIMDVector<S, T, E> predictedLabels, SIMDVector<S, T, E> origLabels);
}
