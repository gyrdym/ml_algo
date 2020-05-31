import 'package:ml_algo/src/common/exception/matrix_column_exception.dart';
import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:xrange/integers.dart';

class LeastSquareCostFunction implements CostFunction {
  const LeastSquareCostFunction();

  @override
  double getCost(Matrix predictedLabels, Matrix originalLabels) {
    if (predictedLabels.columnsNum != 1) {
      throw MatrixColumnException(predictedLabels.rowsNum,
          predictedLabels.columnsNum);
    }

    if (originalLabels.columnsNum != 1) {
      throw MatrixColumnException(originalLabels.rowsNum,
          originalLabels.columnsNum);
    }

    return (predictedLabels - originalLabels)
        .pow(2)
        .sum();
  }

  @override
  Matrix getGradient(Matrix x, Matrix w, Matrix y) =>
      x.transpose() * -2 * (y - x * w);

  @override
  Vector getSubGradient(int j, Matrix x, Matrix w, Matrix y) {
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
    }

    if (column == x.columnsNum - 1) {
      return x.sample(columnIndices: integers(0, column, upperClosed: false));
    }

    return x.sample(columnIndices: [
      ...integers(0, column, upperClosed: false),
      ...integers(column + 1, x.columnsNum, upperClosed: false),
    ]);
  }
}
