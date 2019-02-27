import 'package:ml_linalg/matrix.dart';

abstract class WeightsFinder {
  MLMatrix learnWeights(MLMatrix features, MLMatrix labels,
      MLMatrix initialWeights, bool arePointsNormalized);
}
