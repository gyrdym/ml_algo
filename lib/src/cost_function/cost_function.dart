import 'package:linalg/vector.dart';

abstract class CostFunction {
  double getCost(double predictedLabel, double originalLabel);
  double getPartialDerivative(int wIdx, Vector x, Vector w, double y);
  double getSparseSolutionPartial(int wIdx, Vector x, Vector w, double y);
}
