part of gradient_optimizer_base;

class BGDOptimizerImpl extends GradientOptimizerImpl implements BGDOptimizer {
  @override
  Iterable<int> _getSamplesRange(int totalSamplesCount) => [0, totalSamplesCount];
}
