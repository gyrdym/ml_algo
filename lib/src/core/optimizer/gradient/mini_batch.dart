part of 'package:dart_ml/src/core/implementation.dart';

class _MBGDOptimizerImpl extends _GradientOptimizerImpl {
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
    minWeightsDiff: minWeightsDistance,
    iterationLimit: iterationLimit,
    lambda: alpha,
    argumentIncrement: argumentIncrement
  );

  @override
  Iterable<int> _getBatchRange(int numberOfPoints) => _randomizer.getIntegerInterval(0, numberOfPoints);
}
