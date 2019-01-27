import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/score_to_prob_link_function/link_function.dart';
import 'package:ml_linalg/linalg.dart';

class LogLikelihoodCost<T> implements CostFunction<T> {
  final ScoreToProbLinkFunction<T> linkFunction;

  const LogLikelihoodCost(this.linkFunction);

  @override
  double getCost(double score, double yOrig) {
    throw UnimplementedError();
  }

  @override
  MLVector<T> getGradient(MLMatrix<T> x, MLVector<T> w, MLVector<T> y) =>
      (x.transpose() * (y - (x * w).vectorizedMap(linkFunction))).toVector();

  @override
  double getSparseSolutionPartial(int wIdx, MLVector<T> x, MLVector<T> w, double y) =>
      throw UnimplementedError();
}
