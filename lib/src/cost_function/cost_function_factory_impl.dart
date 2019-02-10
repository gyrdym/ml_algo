import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/cost_function/cost_function_factory.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/cost_function/log_likelihood.dart';
import 'package:ml_algo/src/cost_function/squared.dart';
import 'package:ml_algo/src/default_parameter_values.dart';
import 'package:ml_algo/src/link_function/link_function_type.dart';

class CostFunctionFactoryImpl implements CostFunctionFactory {
  const CostFunctionFactoryImpl();

  @override
  CostFunction squared() => SquaredCost();

  @override
  CostFunction logLikelihood(LinkFunctionType linkFunctionType,
          {Type dtype = DefaultParameterValues.dtype}) =>
      LogLikelihoodCost(linkFunctionType, dtype: dtype);

  @override
  CostFunction fromType(CostFunctionType type,
      {Type dtype = DefaultParameterValues.dtype,
      LinkFunctionType linkFunctionType}) {
    switch (type) {
      case CostFunctionType.logLikelihood:
        return logLikelihood(linkFunctionType, dtype: dtype);
      case CostFunctionType.squared:
        return squared();
      default:
        throw UnsupportedError('Unsupported cost function type - $type');
    }
  }
}
