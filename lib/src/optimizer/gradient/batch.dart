import 'package:dart_ml/src/optimizer/gradient/optimizer.dart';
import 'package:dart_ml/src/optimizer/regularization.dart';

class BGDOptimizer extends GradientOptimizer {
  BGDOptimizer(double learningRate, double minWeightsDistance, int iterationLimit, Regularization regularization)
      : super(learningRate, minWeightsDistance, iterationLimit, regularization);

  @override
  Iterable<int> getSampleRange(int totalSamplesCount) => [0, totalSamplesCount];
}
