part of 'base_impl.dart';

class BGDOptimizerImpl extends GradientOptimizerImpl implements BGDOptimizer {
  @override
  Iterable<int> _getSamplesRange(int totalSamplesCount) => [0, totalSamplesCount];
}
