import 'dart:math' as math;

import 'package:ml_algo/src/common/exception/matrix_column_exception.dart';
import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_linalg/linalg.dart';

class LogLikelihoodCostFunction implements CostFunction {
  LogLikelihoodCostFunction(this._linkFunction);

  final LinkFunction _linkFunction;

  @override
  double getCost(Matrix predictedProbabilities, Matrix originalProbabilities) {
    if (predictedProbabilities.columnsNum != 1) {
      throw MatrixColumnException(predictedProbabilities.rowsNum,
          predictedProbabilities.columnsNum);
    }

    if (originalProbabilities.columnsNum != 1) {
      throw MatrixColumnException(originalProbabilities.rowsNum,
          originalProbabilities.columnsNum);
    }

    return math.log(predictedProbabilities.prod());
  }

  @override
  Matrix getGradient(Matrix x, Matrix w, Matrix y) =>
    x.transpose() * (y - _linkFunction.link(x * w));

  @override
  Vector getSubGradient(int wIdx, Matrix x, Matrix w, Matrix y) =>
      throw UnimplementedError();
}
