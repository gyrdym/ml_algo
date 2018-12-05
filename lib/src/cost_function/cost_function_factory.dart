import 'dart:typed_data';

import 'package:ml_algo/src/cost_function/log_likelihood.dart';
import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/cost_function/squared.dart';
import 'package:ml_algo/src/score_to_prob_link_function/link_function.dart';

class CostFunctionFactory {
  static CostFunction<Float32x4> squared() => const SquaredCost();
  static CostFunction<Float32x4> logLikelihood(VectorizedScoreToProbLinkFunction<Float32x4> linkFn) =>
      LogLikelihoodCost(linkFn);
}
