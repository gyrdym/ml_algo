import 'package:ml_linalg/matrix.dart';

abstract class CostFunction {
  double getCost(Matrix x, Matrix w, Matrix y);
  Matrix getGradient(Matrix x, Matrix w, Matrix y);
}
