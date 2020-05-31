import 'dart:math' as math;

import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_linalg/linalg.dart';

class LogLikelihoodCostFunction implements CostFunction {
  LogLikelihoodCostFunction(
      this._linkFunction,
      this._positiveLabel,
      this._negativeLabel);

  final LinkFunction _linkFunction;
  final num _positiveLabel;
  final num _negativeLabel;

  @override
  double getCost(Matrix x, Matrix w, Matrix y) {
    throw UnimplementedError();
  }

  @override
  Matrix getGradient(Matrix x, Matrix w, Matrix y) =>
    x.transpose() * (y - _linkFunction.link(x * w));

  @override
  Vector getSubGradient(int wIdx, Matrix x, Matrix w, Matrix y) =>
      throw UnimplementedError();
}
