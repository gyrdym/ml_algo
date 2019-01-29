import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';

abstract class CostFunctionFactory {
  CostFunction fromType(CostFunctionType type, {Function scoreToProbLink});
  CostFunction squared();
  CostFunction logLikelihood(Function linkFn);
}
