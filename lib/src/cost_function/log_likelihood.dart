import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_linalg/linalg.dart';

class LogLikelihoodCost implements CostFunction {
  LogLikelihoodCost(this._linkFunction);

  final LinkFunction _linkFunction;

  @override
  double getCost(double score, double yOrig) {
    throw UnimplementedError();
  }

  @override
  Matrix getGradient(Matrix x, Matrix w, Matrix y) =>
    x.transpose() * (y - _linkFunction.link(x * w));

  @override
  Vector getSubDerivative(int wIdx, Matrix x, Matrix w, Matrix y) =>
      throw UnimplementedError();
}
