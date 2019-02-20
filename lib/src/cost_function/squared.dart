import 'dart:math' as math;

import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_linalg/linalg.dart';

class SquaredCost implements CostFunction {
  @override
  double getCost(double predictedLabel, double originalLabel) =>
      math.pow(predictedLabel - originalLabel, 2).toDouble();

  @override
  MLMatrix getGradient(MLMatrix x, MLMatrix w, MLMatrix y) =>
      x.transpose() * -2 * (y - x * w);

  @override
  MLVector getSubDerivative(int j, MLMatrix x, MLMatrix w, MLMatrix y) {
    final xj = x.pick(columnRanges: List<Range>.generate(y.columnsNum,
            (i) => Range(j, j, endInclusive: true)));
    final xWithoutJ = _excludeColumn(x, j);
    final wWithoutJ = _excludeColumn(w, j);
    final predictionWithoutJ = xWithoutJ * wWithoutJ;
    // not a matrix mul!!!
    return (xj * (y - predictionWithoutJ)).reduceRows((sum, row) => sum + row);
  }

  MLMatrix _excludeColumn(MLMatrix x, int column) {
    if (column == 0) {
      return x.submatrix(columns: Range(1, x.columnsNum));
    } else if (column == x.columnsNum - 1) {
      return x.submatrix(columns: Range(0, column));
    } else {
      return x.pick(columnRanges: [Range(0, column), Range(column + 1,
          x.columnsNum)]);
    }
  }
}
