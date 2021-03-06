import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:xrange/xrange.dart';

class LeastSquareCostFunction implements CostFunction {
  const LeastSquareCostFunction();

  @override
  double getCost(Matrix x, Matrix w, Matrix y) => (x * w - y).pow(2).sum();

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
      return x.sample(columnIndices: integers(1, x.columnsNum));
    }

    if (column == x.columnsNum - 1) {
      return x.sample(columnIndices: integers(0, column, upperClosed: false));
    }

    return x.sample(columnIndices: [
      ...integers(0, column),
      ...integers(column + 1, x.columnsNum),
    ]);
  }
}
