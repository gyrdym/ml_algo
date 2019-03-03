import 'package:ml_linalg/matrix.dart';

abstract class WeightsFinder {
  Matrix learnWeights(Matrix features, Matrix labels,
      Matrix initialWeights, bool arePointsNormalized);
}
