import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

abstract class WeightsFinder {
  MLMatrix learnWeights(MLMatrix features, MLVector labels,
      MLMatrix initialWeights, bool arePointsNormalized);
}
