import 'package:dart_ml/src/di/injector.dart';
import 'package:dart_ml/src/math/misc/randomizer/randomizer.dart';
import 'package:dart_ml/src/optimizer/gradient/optimizer.dart';
import 'package:dart_ml/src/optimizer/gradient/interface/mini_batch.dart';

class MBGDOptimizerImpl extends GradientOptimizer implements MBGDOptimizer {
  final Randomizer _randomizer = injector.get(Randomizer);

  @override
  Iterable<int> getSampleRange(int totalSamplesCount) {
    int end = _randomizer.getIntegerFromInterval(1, totalSamplesCount);
    int start = _randomizer.getIntegerFromInterval(0, end);

    return [start, end];
  }
}
