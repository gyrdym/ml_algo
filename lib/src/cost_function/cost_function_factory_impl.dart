import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/cost_function/cost_function_factory.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/cost_function/log_likelihood.dart';
import 'package:ml_algo/src/cost_function/squared.dart';
import 'package:ml_algo/src/utils/default_parameter_values.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper_type.dart';
import 'package:ml_linalg/dtype.dart';

class CostFunctionFactoryImpl implements CostFunctionFactory {
  const CostFunctionFactoryImpl();

  @override
  CostFunction squared() => SquaredCost();

  @override
  CostFunction logLikelihood(ScoreToProbMapperType scoreToProbMapperType,
          {DType dtype = DefaultParameterValues.dtype}) =>
      LogLikelihoodCost(scoreToProbMapperType, dtype: dtype);

  @override
  CostFunction fromType(CostFunctionType type,
      {DType dtype = DefaultParameterValues.dtype,
      ScoreToProbMapperType scoreToProbMapperType}) {
    switch (type) {
      case CostFunctionType.logLikelihood:
        return logLikelihood(scoreToProbMapperType, dtype: dtype);
      case CostFunctionType.squared:
        return squared();
      default:
        throw UnsupportedError('Unsupported cost function type - $type');
    }
  }
}
