import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_linalg/matrix.dart';

/// An interface for all the predicting entities: regressor, classifiers, etc.
abstract class Predictor {
  /// A matrix of training observations, that was used to fit the predictor
  Matrix get trainingFeatures;

  /// A matrix of dependant variables, that was used to fit the predictor
  Matrix get trainingOutcomes;

  /// A flag that shows whether the intercept term is fitted or not
  bool get fitIntercept;

  /// A multiplier of intercept term (meaningful only if [fitIntercept] is true)
  double get interceptScale;

  /// Assesses model according to provided [metric]
  double test(Matrix observations, Matrix outcomes, MetricType metric);
}
