import 'dart:math' as math;
import 'dart:collection';

import 'package:dart_ml/src/enums.dart';
import 'package:dart_ml/src/math/vector/vector_interface.dart';

abstract class Vector extends ListBase<double> implements VectorInterface {
  Vector();
  Vector.from(Iterable<double> source);
  Vector.filled(int length, double value);

  double vectorScalarMult(VectorInterface vector) => (this * vector).sum();
  double distanceTo(VectorInterface vector, [Norm norm = Norm.EUCLIDEAN]) => (this - vector).norm(norm);
  double mean() => sum() / length;
  double norm([Norm norm = Norm.EUCLIDEAN]) {
    int exp;

    switch(norm) {
      case Norm.EUCLIDEAN:
        exp = 2;
        break;
      case Norm.MANHATTAN:
        exp = 1;
        break;
    }

    return math.pow(intPow(exp).abs().sum(), 1 / exp);
  }
}
