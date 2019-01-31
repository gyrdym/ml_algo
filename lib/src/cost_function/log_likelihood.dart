import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_linalg/linalg.dart';

class LogLikelihoodCost implements CostFunction {
  final Function linkFunction;

  const LogLikelihoodCost(this.linkFunction);

  @override
  double getCost(double score, double yOrig) {
    throw UnimplementedError();
  }

  @override
  MLVector getGradient(MLMatrix x, MLVector w, MLVector y) =>
      (x.transpose() * (y - (x * w).fastMap(linkFunction))).toVector();

  @override
  double getSparseSolutionPartial(int wIdx, MLVector x, MLVector w, double y) =>
      throw UnimplementedError();
}
