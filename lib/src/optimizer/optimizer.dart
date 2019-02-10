import 'package:ml_linalg/linalg.dart';

abstract class Optimizer {
  MLVector findExtrema(MLMatrix points, MLVector labels,
      {MLVector initialWeights, bool isMinimizingObjective, bool arePointsNormalized});
}
