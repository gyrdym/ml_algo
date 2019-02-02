import 'dart:typed_data';

import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/cost_function/cost_function_factory.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/cost_function/log_likelihood.dart';
import 'package:ml_algo/src/cost_function/squared.dart';
import 'package:ml_algo/src/link_function/link_function.dart';

class CostFunctionFactoryImpl implements CostFunctionFactory {
  const CostFunctionFactoryImpl();

  @override
  CostFunction squared() => SquaredCost();

  @override
  CostFunction logLikelihood(LinkFunction linkFunction, {Type dtype = Float32x4}) =>
      LogLikelihoodCost(linkFunction, dtype: dtype);

  @override
  CostFunction fromType(CostFunctionType type, {Type dtype = Float32x4, LinkFunction linkFunction}) {
    switch (type) {
      case CostFunctionType.logLikelihood:
        return logLikelihood(linkFunction, dtype: dtype);
      case CostFunctionType.squared:
        return squared();
      default:
        throw UnsupportedError('Unsupported cost function type - $type');
    }
  }
}
