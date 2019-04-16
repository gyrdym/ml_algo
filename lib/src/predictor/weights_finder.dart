import 'package:ml_linalg/matrix.dart';

abstract class WeightsFinder {
  Matrix learnWeights(Matrix observations, Matrix outcomes,
      Matrix initialWeights, bool arePointsNormalized);
}
