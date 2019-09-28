import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/model_selection/assessable.dart';
import 'package:ml_algo/src/model_selection/cross_validator/cross_validator_impl.dart';
import 'package:ml_algo/src/model_selection/data_splitter/k_fold.dart';
import 'package:ml_algo/src/model_selection/data_splitter/leave_p_out.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';

typedef PredictorFactory = Assessable Function(DataFrame observations,
    Iterable<String> targetNames);

/// A factory and an interface for all the cross validator types
abstract class CrossValidator {
  /// Creates k-fold validator to evaluate quality of a predictor.
  ///
  /// It splits a dataset into [numberOfFolds] test sets and subsequently
  /// evaluates the predictor on each produced test set
  factory CrossValidator.kFold(
      DataFrame samples,
      Iterable<String> targetNames, {
        int numberOfFolds = 5,
        DType dtype = DType.float32,
      }) =>
      CrossValidatorImpl(
        samples,
        targetNames,
        KFoldSplitter(numberOfFolds),
        dtype,
      );

  /// Creates LPO validator to evaluate quality of a predictor.
  ///
  /// It splits a dataset into all possible test sets of size [p] and
  /// subsequently evaluates quality of the predictor on each produced test set
  factory CrossValidator.lpo(
      DataFrame samples,
      Iterable<String> targetNames,
      int p, {
        DType dtype = DType.float32,
      }) =>
      CrossValidatorImpl(
        samples,
        targetNames,
        LeavePOutSplitter(p),
        dtype,
      );

  /// Returns a score of quality of passed predictor depending on given
  /// [metricType]
  double evaluate(PredictorFactory predictorFactory, MetricType metricType);
}
