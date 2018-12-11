import 'package:ml_linalg/linalg.dart';

abstract class CostFunction<E> {
  double getCost(double predictedLabel, double originalLabel);

  MLVector<E> getGradient(MLMatrix<E> x, MLVector<E> w, MLVector<E> y);

  double getSparseSolutionPartial(int wIdx, MLVector<E> x, MLVector<E> w, double y);
}
