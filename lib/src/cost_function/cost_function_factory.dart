import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/link_function/link_function.dart';

abstract class CostFunctionFactory {
  CostFunction createByType(CostFunctionType type, {
    LinkFunction linkFunction,
  });
}
