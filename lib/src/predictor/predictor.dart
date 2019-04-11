import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

/// An interface for all the predicting entities: regressor, classifiers, etc.
abstract class Predictor {
  /// Learned coefficients (or weights) for given features
  Vector get weights;

  /// Fits the passed [observations] to true labels - [outcomes]. It's
  /// possible to provide [initialWeights] and specify, whether the [observations]
  /// normalized or not
  void fit(Matrix observations, Matrix outcomes,
      {Matrix initialWeights, bool isDataNormalized});

  /// Assesses model according to provided [metric]
  double test(Matrix observations, Matrix outcomes, MetricType metric);
}
