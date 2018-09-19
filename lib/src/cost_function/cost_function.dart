import 'dart:typed_data';

import 'package:linalg/vector.dart';

abstract class CostFunction<S extends List<E>, T extends List<double>, E> {
  double getCost(double predictedLabel, double originalLabel);
  double getPartialDerivative(
    int wIdx,
    SIMDVector<S, T, E> x,
    SIMDVector<S, T, E> w,
    double y
  );
  double getSparseSolutionPartial(
    int wIdx,
    SIMDVector<S, T, E> x,
    SIMDVector<S, T, E> w,
    double y
  );
}
