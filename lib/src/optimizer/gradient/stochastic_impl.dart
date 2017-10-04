part of 'package:dart_ml/src/implementation.dart';

class _SGDOptimizerImpl extends _GradientOptimizerImpl implements Optimizer {
  final Randomizer _randomizer = injector.get(Randomizer);

  _SGDOptimizerImpl({
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
  Iterable<int> _getSamplesRange(int totalSamplesCount) {
    int k = _randomizer.getIntegerFromInterval(0, totalSamplesCount);
    return [k, k + 1];
  }
}
