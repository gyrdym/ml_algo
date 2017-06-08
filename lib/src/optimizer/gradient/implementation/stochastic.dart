import 'package:dart_ml/src/di/injector.dart';
import 'package:dart_ml/src/math/misc/randomizer/interface/randomizer.dart';
import 'package:dart_ml/src/optimizer/gradient/optimizer.dart';
import 'package:dart_ml/src/optimizer/gradient/interface/stochastic.dart';

class SGDOptimizerImpl extends GradientOptimizer implements SGDOptimizer {
  final Randomizer _random = injector.get(Randomizer);

  @override
  Iterable<int> getSampleRange(int totalSamplesCount) {
    int k = _random.getIntegerFromInterval(0, totalSamplesCount);
    return [k, k + 1];
  }
}
