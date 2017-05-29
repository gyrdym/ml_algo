import 'dart:math' as math;
import 'package:dart_ml/src/optimizer/gradient/base_optimizer.dart';

class MBGDOptimizer extends GradientOptimizer {
  final math.Random _random = new math.Random();

  MBGDOptimizer(double learningRate, double minWeightsDistance, int iterationLimit) : super(
      learningRate,
      minWeightsDistance,
      iterationLimit
    );

  @override
  Iterable<int> getSampleRange(int totalSamplesCount) {
    int end = _random.nextInt(totalSamplesCount - 1) + 1;
    int start = _random.nextInt(end);

    return [start, end];
  }
}
