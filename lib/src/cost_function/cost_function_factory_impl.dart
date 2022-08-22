import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/cost_function/cost_function_factory.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/cost_function/least_square_cost_function.dart';
import 'package:ml_algo/src/cost_function/log_likelihood_cost_function.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_linalg/dtype.dart';

class CostFunctionFactoryImpl implements CostFunctionFactory {
  const CostFunctionFactoryImpl();

  @override
  CostFunction createByType(
    CostFunctionType type, {
    LinkFunction? linkFunction,
    num? positiveLabel,
    num? negativeLabel,
    DType dtype = DType.float32,
  }) {
    switch (type) {
      case CostFunctionType.logLikelihood:
        if (linkFunction == null) {
          throw Exception('Link function must be specified if log likelihood '
              'cost function is going to be used');
        }

        if (positiveLabel == null) {
          throw Exception('Positive label must be specified');
        }

        if (negativeLabel == null) {
          throw Exception('Negative label must be specified');
        }

        return LogLikelihoodCostFunction(
          linkFunction,
          positiveLabel,
          negativeLabel,
          dtype,
        );

      case CostFunctionType.leastSquare:
        return const LeastSquareCostFunction();

      default:
        throw UnsupportedError('Unsupported cost function type - $type');
    }
  }
}
