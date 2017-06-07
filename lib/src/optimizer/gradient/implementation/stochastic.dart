import 'dart:math' as math;

import 'package:dart_ml/src/optimizer/gradient/optimizer.dart';
import 'package:dart_ml/src/optimizer/gradient/interface/stochastic.dart';

class SGDOptimizerImpl extends GradientOptimizer implements SGDOptimizer {
  final math.Random _random = new math.Random();

  @override
  Iterable<int> getSampleRange(int totalSamplesCount) {
    int k = _random.nextInt(totalSamplesCount);
    return [k, k + 1];
  }
}
