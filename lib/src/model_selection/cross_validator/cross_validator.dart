import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/predictor/predictor.dart';
import 'package:ml_algo/src/model_selection/cross_validator/cross_validator_impl.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

/// A factory and an interface for all the cross validator types
abstract class CrossValidator {
  /// Creates k-fold validator to evaluate quality of a predictor.
  ///
  /// It splits a dataset into [numberOfFolds] test sets and subsequently
  /// evaluates the predictor on each produced test set
  factory CrossValidator.kFold({Type dtype, int numberOfFolds}) =
      CrossValidatorImpl.kFold;

  /// Creates LPO validator to evaluate quality of a predictor.
  ///
  /// It splits a dataset into all possible test sets of size [p] and
  /// subsequently evaluates quality of the predictor on each produced test set
  factory CrossValidator.lpo({Type dtype, int p}) = CrossValidatorImpl.lpo;

  /// Returns a score of quality of passed predictor depending on given [metric]
  double evaluate(
      Predictor predictor, MLMatrix points, MLVector labels, MetricType metric,
      {bool isDataNormalized = false});
}
