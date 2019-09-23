import 'dart:math' as math;

import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:xrange/integers.dart';

class SquaredCost implements CostFunction {
  const SquaredCost();

  @override
  double getCost(double predictedLabel, double originalLabel) =>
      math.pow(predictedLabel - originalLabel, 2).toDouble();

  @override
  Matrix getGradient(Matrix x, Matrix w, Matrix y) =>
      x.transpose() * -2 * (y - x * w);

  @override
  Vector getSubDerivative(int j, Matrix x, Matrix w, Matrix y) {
    final xj = x.sample(columnIndices: List.generate(y.columnsNum, (_) => j));
    final xWithoutJ = _excludeColumn(x, j);
    final wWithoutJ = _excludeColumn(w, j);
    final predictionWithoutJ = xWithoutJ * wWithoutJ.transpose();
    return (xj.transpose() * (y - predictionWithoutJ))
        .reduceRows((sum, row) => sum + row);
  }

  Matrix _excludeColumn(Matrix x, int column) {
    if (column == 0) {
      return x.sample(columnIndices: integers(1, x.columnsNum,
          upperClosed: false));
    } else if (column == x.columnsNum - 1) {
      return x.sample(columnIndices: integers(0, column, upperClosed: false));
    } else {
      return x.sample(columnIndices: [
        ...integers(0, column, upperClosed: false),
        ...integers(column + 1, x.columnsNum, upperClosed: false),
      ]);
    }
  }
}
