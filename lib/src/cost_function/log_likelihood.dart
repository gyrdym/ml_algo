import 'dart:typed_data';

import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/default_parameter_values.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/link_function/link_function_factory.dart';
import 'package:ml_algo/src/link_function/link_function_factory_impl.dart';
import 'package:ml_algo/src/link_function/link_function_type.dart';
import 'package:ml_linalg/linalg.dart';

class LogLikelihoodCost implements CostFunction {
  final LinkFunction linkFunction;
  final Type dtype;

  LogLikelihoodCost(
    LinkFunctionType linkFunctionType, {
    this.dtype = DefaultParameterValues.dtype,
    LinkFunctionFactory linkFunctionFactory = const LinkFunctionFactoryImpl(),
  }) : linkFunction = linkFunctionFactory.fromType(linkFunctionType, dtype);

  @override
  double getCost(double score, double yOrig) {
    throw UnimplementedError();
  }

  @override
  MLVector getGradient(MLMatrix x, MLVector w, MLVector y) {
    final scores = (x * w).toVector();
    switch (dtype) {
      case Float32x4:
        return (x.transpose() * (y - linkFunction.linkScoresToProbs(scores)))
            .toVector();
      default:
        throw throw UnsupportedError('Unsupported data type - $dtype');
    }
  }

  @override
  double getSparseSolutionPartial(int wIdx, MLVector x, MLVector w, double y) =>
      throw UnimplementedError();
}
