import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

abstract class CostFunction {
  double getCost(double predictedLabel, double originalLabel);
  Matrix getGradient(Matrix x, Matrix w, Matrix y);
  Vector getSubDerivative(int j, Matrix x, Matrix w, Matrix y);
}
