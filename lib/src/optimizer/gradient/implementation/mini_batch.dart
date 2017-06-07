import 'dart:math' as math;

import 'package:dart_ml/src/optimizer/gradient/optimizer.dart';
import 'package:dart_ml/src/optimizer/gradient/interface/mini_batch.dart';

class MBGDOptimizerImpl extends GradientOptimizer implements MBGDOptimizer {
  final math.Random _random = new math.Random();

  @override
  Iterable<int> getSampleRange(int totalSamplesCount) {
    int end = _random.nextInt(totalSamplesCount - 1) + 1;
    int start = _random.nextInt(end);

    return [start, end];
  }
}
