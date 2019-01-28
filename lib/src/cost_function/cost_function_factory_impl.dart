import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/cost_function/cost_function_factory.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/cost_function/log_likelihood.dart';
import 'package:ml_algo/src/cost_function/squared.dart';
import 'package:ml_algo/src/score_to_prob_link_function/link_function.dart';

class CostFunctionFactoryImpl implements CostFunctionFactory {
  const CostFunctionFactoryImpl();

  @override
  CostFunction<T> squared<T>() => SquaredCost<T>();

  @override
  CostFunction<T> logLikelihood<T>(ScoreToProbLinkFunction<T> linkFn) =>
      LogLikelihoodCost(linkFn);

  @override
  CostFunction<T> fromType<T>(CostFunctionType type, {ScoreToProbLinkFunction<T> scoreToProbLink}) {
    switch (type) {
      case CostFunctionType.logLikelihood:
        return logLikelihood<T>(scoreToProbLink);
      case CostFunctionType.squared:
        return squared<T>();
      default:
        throw UnimplementedError();
    }
  }
}
