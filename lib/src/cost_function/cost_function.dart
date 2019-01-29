import 'package:ml_linalg/linalg.dart';

abstract class CostFunction {
  double getCost(double predictedLabel, double originalLabel);
  MLVector getGradient(MLMatrix x, MLVector w, MLVector y);
  double getSparseSolutionPartial(int wIdx, MLVector x, MLVector w, double y);
}
