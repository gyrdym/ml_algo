import 'package:dart_ml/src/optimizer/gradient/implementation/base.dart';
import 'package:dart_ml/src/optimizer/gradient/interface/batch.dart';

class BGDOptimizerImpl extends GradientOptimizerImpl implements BGDOptimizer {
  @override
  Iterable<int> getSampleRange(int totalSamplesCount) => [0, totalSamplesCount];
}
