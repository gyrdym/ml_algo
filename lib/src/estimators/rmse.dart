import 'dart:math' as math;

import 'package:dart_ml/src/estimators/estimator.dart';
import 'package:dart_ml/src/math/typed_vector.dart' as vectors;

class RMSE implements Estimator {
  double calculateError(List<double> predictedLabels, List<double> origLabels) {
    List<double> diffs = vectors.subtraction(predictedLabels, origLabels);
    List<double> errors = vectors.pow(diffs, 2);

    return math.sqrt(vectors.mean(errors));
  }
}