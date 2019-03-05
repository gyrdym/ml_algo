import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

/// An interface for all the predicting entities: regressor, classifiers, etc.
abstract class Predictor {
  /// Learned coefficients (or weights) for given features
  Vector get weights;

  /// Fits the passed [features] to true labels - [origLabels]. It's
  /// possible to provide [initialWeights] and specify, whether the [features]
  /// normalized or not
  void fit(Matrix features, Matrix origLabels,
      {Matrix initialWeights, bool isDataNormalized});

  /// Assesses model according to provided [metric]
  double test(Matrix features, Matrix origLabels, MetricType metric);
}
