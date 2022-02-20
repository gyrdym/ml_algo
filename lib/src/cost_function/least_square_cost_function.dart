import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_linalg/linalg.dart';

class LeastSquareCostFunction implements CostFunction {
  const LeastSquareCostFunction();

  @override
  double getCost(Matrix x, Matrix w, Matrix y) => (x * w - y).pow(2).sum();

  @override
  Matrix getGradient(Matrix x, Matrix w, Matrix y) =>
      x.transpose() * -2 * (y - x * w);

  @override
  Vector getSubGradient(int j, Matrix X, Matrix W, Matrix Y) {
    final xj = X.getColumn(j);
    final XWithoutJ = X.filterColumns((column, idx) => idx != j);
    final WWithoutJ = W.filterColumns((column, idx) => idx != j);
    final predictionWithoutJ = XWithoutJ * WWithoutJ.toVector();

    return Vector.fromList([(xj * (Y - predictionWithoutJ).toVector()).sum()]);
  }
}
