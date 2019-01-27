import 'package:ml_linalg/linalg.dart';

abstract class CostFunction<T> {
  double getCost(double predictedLabel, double originalLabel);
  MLVector<T> getGradient(MLMatrix<T> x, MLVector<T> w, MLVector<T> y);
  double getSparseSolutionPartial(int wIdx, MLVector<T> x, MLVector<T> w, double y);
}
