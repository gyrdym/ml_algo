import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/score_to_prob_link_function/link_function.dart';

abstract class CostFunctionFactory {
  CostFunction<T> fromType<T>(CostFunctionType type, {ScoreToProbLinkFunction<T> scoreToProbLink});
  CostFunction<T> squared<T>();
  CostFunction<T> logLikelihood<T>(ScoreToProbLinkFunction<T> linkFn);
}
