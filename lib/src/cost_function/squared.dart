import 'dart:math' as math;

import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:xrange/zrange.dart';

class SquaredCost implements CostFunction {
  @override
  double getCost(double predictedLabel, double originalLabel) =>
      math.pow(predictedLabel - originalLabel, 2).toDouble();

  @override
  Matrix getGradient(Matrix x, Matrix w, Matrix y) =>
      x.transpose() * -2 * (y - x * w);

  @override
  Vector getSubDerivative(int j, Matrix x, Matrix w, Matrix y) {
    final xj = x.pick(columnRanges: List.generate(y.columnsNum,
            (i) => ZRange.singleton(j)));
    final xWithoutJ = _excludeColumn(x, j);
    final wWithoutJ = _excludeColumn(w, j);
    final predictionWithoutJ = xWithoutJ * wWithoutJ.transpose();
    return (xj.transpose() * (y - predictionWithoutJ))
        .reduceRows((sum, row) => sum + row);
  }

  Matrix _excludeColumn(Matrix x, int column) {
    if (column == 0) {
      return x.submatrix(columns: ZRange.closedOpen(1, x.columnsNum));
    } else if (column == x.columnsNum - 1) {
      return x.submatrix(columns: ZRange.closedOpen(0, column));
    } else {
      return x.pick(columnRanges: [
        ZRange.closedOpen(0, column),
        ZRange.closedOpen(column + 1, x.columnsNum)
      ]);
    }
  }
}
