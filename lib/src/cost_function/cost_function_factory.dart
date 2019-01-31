import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/link_function/link_function.dart';

abstract class CostFunctionFactory {
  CostFunction fromType(CostFunctionType type, {Type dtype, LinkFunction linkFunction});
  CostFunction squared();
  CostFunction logLikelihood(LinkFunction linkFunction, {Type dtype});
}
