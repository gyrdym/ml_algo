import 'package:ml_linalg/linalg.dart';

abstract class CostFunction<E> {
  double getCost(double predictedLabel, double originalLabel);
  double getPartialDerivative(int wIdx, MLVector<E> x, MLVector<E> w, double y);
  MLMatrix<E, MLVector<E>> getGradient(MLMatrix<E, MLVector<E>> x, MLMatrix<E, MLVector<E>> w, MLMatrix<E, MLVector<E>> y);
  double getSparseSolutionPartial(int wIdx, MLVector<E> x, MLVector<E> w, double y);
}
