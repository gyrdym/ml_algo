import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper_type.dart';
import 'package:ml_linalg/dtype.dart';

abstract class CostFunctionFactory {
  CostFunction fromType(CostFunctionType type,
      {DType dtype, ScoreToProbMapperType scoreToProbMapperType});
  CostFunction squared();
  CostFunction logLikelihood(ScoreToProbMapperType scoreToProbMapperType,
      {DType dtype});
}
