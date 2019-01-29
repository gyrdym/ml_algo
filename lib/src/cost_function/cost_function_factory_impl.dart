import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/cost_function/cost_function_factory.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/cost_function/log_likelihood.dart';
import 'package:ml_algo/src/cost_function/squared.dart';

class CostFunctionFactoryImpl implements CostFunctionFactory {
  const CostFunctionFactoryImpl();

  @override
  CostFunction squared() => SquaredCost();

  @override
  CostFunction logLikelihood(Function linkFn) => LogLikelihoodCost(linkFn);

  @override
  CostFunction fromType(CostFunctionType type, {Function scoreToProbLink}) {
    switch (type) {
      case CostFunctionType.logLikelihood:
        return logLikelihood(scoreToProbLink);
      case CostFunctionType.squared:
        return squared();
      default:
        throw UnimplementedError();
    }
  }
}
