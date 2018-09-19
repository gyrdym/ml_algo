import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_ml/src/cost_function/cost_function.dart';
import 'package:dart_ml/src/score_to_prob_link_function/link_function.dart' as linkFunctions;
import 'package:linalg/vector.dart';

class LogLikelihoodCost implements CostFunction<Float32x4List, Float32List, Float32x4> {
  const LogLikelihoodCost();

  @override
  double getCost(double score, double yOrig) {
    final probability = linkFunctions.logitLink(score);
    return _indicator(yOrig, -1.0) * math.log(1 - probability) + _indicator(yOrig, 1.0) * math.log(probability);
  }

  @override
  double getPartialDerivative(
    int idx,
    SIMDVector x,
    SIMDVector w,
    double y
  ) =>  x[idx] * (_indicator(y, 1.0) - linkFunctions.logitLink(x.dot(w)));

  int _indicator(double y, double target) => target == y ? 1 : 0;

  @override
  double getSparseSolutionPartial(
    int wIdx,
    SIMDVector x,
    SIMDVector w,
    double y
  ) => throw UnimplementedError();
}
