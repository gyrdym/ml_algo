part of gradient_optimizer_base;

class MBGDOptimizerImpl extends GradientOptimizerImpl implements MBGDOptimizer {
  final Randomizer _randomizer = injector.get(Randomizer);

  @override
  Iterable<int> _getSamplesRange(int totalSamplesCount) {
    int end = _randomizer.getIntegerFromInterval(1, totalSamplesCount);
    int start = _randomizer.getIntegerFromInterval(0, end);

    return [start, end];
  }
}
