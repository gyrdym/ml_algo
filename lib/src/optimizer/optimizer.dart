import 'package:ml_linalg/linalg.dart';

abstract class Optimizer<E, T extends MLVector<E>> {
  MLVector<E> findExtrema(MLMatrix<E, T> points, MLVector<E> labels,
      {MLVector<E> initialWeights, bool isMinimizingObjective, bool arePointsNormalized});
}
