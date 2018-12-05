import 'dart:math' as math;
import 'dart:typed_data';

import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/score_to_prob_link_function/link_function_impl.dart' as linkFunctions;
import 'package:ml_linalg/linalg.dart';

class LogLikelihoodCost implements CostFunction<Float32x4> {
  const LogLikelihoodCost();

  @override
  double getCost(double score, double yOrig) {
    final probability = linkFunctions.logitLink(score);
    return _indicator(yOrig, -1.0) * math.log(1 - probability) + _indicator(yOrig, 1.0) * math.log(probability);
  }

  @override
  double getPartialDerivative(int idx, MLVector<Float32x4> x, MLVector<Float32x4> w, double y) =>
      x[idx] * (_indicator(y, 1.0) - linkFunctions.logitLink(x.dot(w)));

  int _indicator(double y, double target) => target == y ? 1 : 0;

  @override
  MLMatrix<Float32x4, MLVector<Float32x4>> getGradient(
    MLMatrix<Float32x4, MLVector<Float32x4>> x,
    MLMatrix<Float32x4, MLVector<Float32x4>> w,
    MLMatrix<Float32x4, MLVector<Float32x4>> y
  ) => x.transpose() *
      (y.mapColumns(linkFunctions.vectorizedIndicator) - (x * w).mapColumns(linkFunctions.vectorizedLogitLink));

  @override
  double getSparseSolutionPartial(int wIdx, MLVector<Float32x4> x, MLVector<Float32x4> w, double y) =>
      throw UnimplementedError();
}
