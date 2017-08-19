part of 'package:dart_ml/src/dart_ml_impl.dart';

class MBGDOptimizerImpl extends _GradientOptimizerImpl implements MBGDOptimizer {
  final Randomizer _randomizer = injector.get(Randomizer);

  @override
  Iterable<int> _getSamplesRange(int totalSamplesCount) {
    int end = _randomizer.getIntegerFromInterval(1, totalSamplesCount);
    int start = _randomizer.getIntegerFromInterval(0, end);

    return [start, end];
  }
}
