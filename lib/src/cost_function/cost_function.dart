import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

abstract class CostFunction {
  double getCost(Matrix predictedLabels, Matrix originalLabel);
  Matrix getGradient(Matrix x, Matrix w, Matrix y);
  Vector getSubGradient(int j, Matrix x, Matrix w, Matrix y);
}
