import 'package:dart_ml/src/optimizer/gradient/optimizer.dart';
import 'package:dart_ml/src/optimizer/gradient/interface/batch.dart';

class BGDOptimizerImpl extends GradientOptimizer implements BGDOptimizer {
  @override
  Iterable<int> getSampleRange(int totalSamplesCount) => [0, totalSamplesCount];
}
