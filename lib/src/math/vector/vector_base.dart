import 'dart:math' as math;
import 'dart:collection';

import 'package:dart_ml/src/math/vector/norm.dart';
import 'package:dart_ml/src/math/vector/vector.dart';

abstract class VectorBase extends ListBase<double> implements Vector {
  double dot(Vector vector) => (this * vector).sum();
  double distanceTo(Vector vector, [Norm norm = Norm.EUCLIDEAN]) => (this - vector).norm(norm);
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
