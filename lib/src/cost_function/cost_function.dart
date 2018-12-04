import 'package:ml_linalg/linalg.dart';

abstract class CostFunction<E> {
  double getCost(double predictedLabel, double originalLabel);
  double getPartialDerivative(int wIdx, MLVector<E> x, MLVector<E> w, double y);
  double getSparseSolutionPartial(int wIdx, MLVector<E> x, MLVector<E> w, double y);
}
