part of 'package:dart_ml/src/implementation.dart';

class _MBGDOptimizerImpl extends _GradientOptimizerImpl implements MBGDOptimizer {
  final Randomizer _randomizer = injector.get(Randomizer);

  @override
  Iterable<int> _getSamplesRange(int totalSamplesCount) => _randomizer.getIntegerInterval(0, totalSamplesCount);
}
