part of 'package:dart_ml/src/dart_ml_impl.dart';

class _BGDOptimizerImpl extends _GradientOptimizerImpl implements BGDOptimizer {
  @override
  Iterable<int> _getSamplesRange(int totalSamplesCount) => [0, totalSamplesCount];
}
