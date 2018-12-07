import 'package:ml_linalg/linalg.dart';

abstract class Optimizer<E> {
  MLVector<E> findExtrema(MLMatrix<E> points, MLVector<E> labels,
      {MLVector<E> initialWeights, bool isMinimizingObjective, bool arePointsNormalized});
}
