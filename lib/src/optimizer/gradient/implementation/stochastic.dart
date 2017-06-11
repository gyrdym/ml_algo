import 'package:dart_ml/src/di/injector.dart';
import 'package:dart_ml/src/math/misc/randomizer/randomizer.dart';
import 'package:dart_ml/src/optimizer/gradient/implementation/base.dart';
import 'package:dart_ml/src/optimizer/gradient/interface/stochastic.dart';

class SGDOptimizerImpl extends GradientOptimizerImpl implements SGDOptimizer {
  final Randomizer _random = injector.get(Randomizer);

  @override
  Iterable<int> getSampleRange(int totalSamplesCount) {
    int k = _random.getIntegerFromInterval(0, totalSamplesCount);
    return [k, k + 1];
  }
}
