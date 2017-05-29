import 'package:dart_ml/src/optimizer/gradient/base_optimizer.dart';

class BGDOptimizer extends GradientOptimizer {
  BGDOptimizer(double learningRate, double minWeightsDistance, int iterationLimit) : super(
      learningRate,
      minWeightsDistance,
      iterationLimit
    );

  @override
  Iterable<int> getSampleRange(int totalSamplesCount) => [0, totalSamplesCount];
}
