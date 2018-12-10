import 'dart:math' as math;
import 'dart:typed_data';

import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/score_to_prob_link_function/link_function.dart';
import 'package:ml_algo/src/score_to_prob_link_function/link_function_impl.dart' as linkFunctions;
import 'package:ml_linalg/linalg.dart';

class LogLikelihoodCost implements CostFunction<Float32x4> {
  final VectorizedScoreToProbLinkFunction<Float32x4> linkFunction;

  const LogLikelihoodCost(this.linkFunction);

  @override
  double getCost(double score, double yOrig) {
    final probability = linkFunctions.logitLink(score);
    return _indicator(yOrig, -1.0) * math.log(1 - probability) + _indicator(yOrig, 1.0) * math.log(probability);
  }

  int _indicator(double y, double target) => target == y ? 1 : 0;

  @override
  MLVector<Float32x4> getGradient(MLMatrix<Float32x4> x, MLVector<Float32x4> w, MLVector<Float32x4> y) {
    final indicatorFn = (Float32x4 labels) => linkFunctions.vectorizedIndicator(labels, linkFunctions.ones);
    return  (x.transpose() * (y.vectorizedMap(indicatorFn) - (x * w).mapColumns(linkFunction))).toVector();
  }

  @override
  double getSparseSolutionPartial(int wIdx, MLVector<Float32x4> x, MLVector<Float32x4> w, double y) =>
      throw UnimplementedError();
}
