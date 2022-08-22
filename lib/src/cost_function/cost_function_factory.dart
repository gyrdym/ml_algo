import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_linalg/dtype.dart';

abstract class CostFunctionFactory {
  CostFunction createByType(
    CostFunctionType type, {
    LinkFunction? linkFunction,
    num? positiveLabel,
    num? negativeLabel,
    DType dtype,
  });
}
