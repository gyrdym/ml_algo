part of 'package:dart_ml/src/implementation.dart';

class _BGDOptimizerImpl extends _GradientOptimizerImpl implements BGDOptimizer {
  @override
  Iterable<int> _getSamplesRange(int totalSamplesCount) => [0, totalSamplesCount];
}
