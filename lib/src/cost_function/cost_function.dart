import 'package:ml_linalg/linalg.dart';

abstract class CostFunction<E> {
  double getCost(double predictedLabel, double originalLabel);
  double getPartialDerivative(int wIdx, MLVector<E> x, MLVector<E> w, double y);
  MLMatrix<E> getGradient(MLMatrix<E> x, MLMatrix<E> w, MLMatrix<E> y);
  double getSparseSolutionPartial(int wIdx, MLVector<E> x, MLVector<E> w, double y);
}
