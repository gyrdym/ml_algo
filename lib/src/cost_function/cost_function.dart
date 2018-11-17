import 'package:linalg/vector.dart';

abstract class CostFunction<E> {
  double getCost(double predictedLabel, double originalLabel);
  double getPartialDerivative(int wIdx, Vector<E> x, Vector<E> w, double y);
  double getSparseSolutionPartial(int wIdx, Vector<E> x, Vector<E> w, double y);
}
