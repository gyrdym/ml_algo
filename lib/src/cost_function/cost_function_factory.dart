import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/cost_function/log_likelihood.dart';
import 'package:ml_algo/src/cost_function/squared.dart';
import 'package:ml_algo/src/score_to_prob_link_function/link_function.dart';

class CostFunctionFactory {
  static CostFunction squared<T>() => SquaredCost<T>();
  static CostFunction<T> logLikelihood<T>(ScoreToProbLinkFunction<T> linkFn) =>
      LogLikelihoodCost(linkFn);
}
