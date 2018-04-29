import 'package:simd_vector/vector.dart';

abstract class CostFunction {
  double getCost(double predictedLabel, double originalLabel);
  double getPartialDerivative(int wIdx, Vector x, Vector w, double y);
}
