import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/link_function/link_function_type.dart';

abstract class CostFunctionFactory {
  CostFunction fromType(CostFunctionType type,
      {Type dtype, LinkFunctionType linkFunctionType});
  CostFunction squared();
  CostFunction logLikelihood(LinkFunctionType linkFunctionType, {Type dtype});
}
