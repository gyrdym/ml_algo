part of 'package:dart_ml/src/core/implementation.dart';

class _SGDOptimizerImpl extends _GradientOptimizerImpl {
  final Randomizer _randomizer = coreInjector.get(Randomizer);

  _SGDOptimizerImpl({
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
  Iterable<int> _getBatchRange(int numberOfPoints) {
    int k = _randomizer.getIntegerFromInterval(0, numberOfPoints);
    return [k, k + 1];
  }
}
