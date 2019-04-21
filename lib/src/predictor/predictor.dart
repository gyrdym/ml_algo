import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_linalg/matrix.dart';

/// An interface for all the predicting entities: regressor, classifiers, etc.
abstract class Predictor {
  /// A matrix of training observations, that was used to fit the predictor
  Matrix get trainingFeatures;

  /// A matrix of dependant variables, that was used to fit the predictor
  Matrix get trainingOutcomes;

  /// Fits the passed [observations] to true labels - [outcomes]. It's
  /// possible to provide [initialWeights] and specify, whether the [observations]
  /// normalized or not
  void fit({Matrix initialWeights});

  /// Assesses model according to provided [metric]
  double test(Matrix observations, Matrix outcomes, MetricType metric);
}
