part of gradient_optimizer_base;

class SGDOptimizerImpl extends GradientOptimizerImpl implements SGDOptimizer {
  final Randomizer _random = injector.get(Randomizer);

  @override
  Iterable<int> _getSamplesRange(int totalSamplesCount) {
    int k = _random.getIntegerFromInterval(0, totalSamplesCount);
    return [k, k + 1];
  }
}
