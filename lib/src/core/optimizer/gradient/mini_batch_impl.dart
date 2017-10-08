part of 'package:dart_ml/src/core/implementation.dart';

class _MBGDOptimizerImpl extends _GradientOptimizerImpl implements MBGDOptimizer {
  final Randomizer _randomizer = coreInjector.get(Randomizer);

  _MBGDOptimizerImpl({
    double learningRate,
    double minWeightsDistance,
    int iterationLimit,
    Regularization regularization,
    double alpha,
    double argumentIncrement
  }) : super(
    learningRate: learningRate,
    minWeightsDistance: minWeightsDistance,
    iterationLimit: iterationLimit,
    regularization: regularization,
    alpha: alpha,
    argumentIncrement: argumentIncrement
  );

  @override
  Iterable<int> _getSamplesRange(int totalSamplesCount) => _randomizer.getIntegerInterval(0, totalSamplesCount);
}
