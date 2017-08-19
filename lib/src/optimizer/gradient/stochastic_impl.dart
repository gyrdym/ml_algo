part of 'package:dart_ml/src/dart_ml_impl.dart';

class SGDOptimizerImpl extends _GradientOptimizerImpl implements SGDOptimizer {
  final Randomizer _randomizer = injector.get(Randomizer);

  @override
  Iterable<int> _getSamplesRange(int totalSamplesCount) {
    int k = _randomizer.getIntegerFromInterval(0, totalSamplesCount);
    return [k, k + 1];
  }
}
