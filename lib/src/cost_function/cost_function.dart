import 'package:ml_linalg/linalg.dart';

abstract class CostFunction {
  double getCost(double predictedLabel, double originalLabel);
  MLMatrix getGradient(MLMatrix x, MLMatrix w, MLMatrix y);
  MLVector getSubDerivative(int j, MLMatrix x, MLMatrix w, MLMatrix y);
}
