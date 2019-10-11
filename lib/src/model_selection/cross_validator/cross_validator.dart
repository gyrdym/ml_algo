import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/model_selection/assessable.dart';
import 'package:ml_algo/src/model_selection/cross_validator/cross_validator_impl.dart';
import 'package:ml_algo/src/model_selection/data_splitter/k_fold.dart';
import 'package:ml_algo/src/model_selection/data_splitter/leave_p_out.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/linalg.dart';

typedef PredictorFactory = Assessable Function(DataFrame observations,
    Iterable<String> targetNames);

typedef DataPreprocessFn = List<DataFrame> Function(DataFrame trainData,
    DataFrame testData);

/// A factory and an interface for all the cross validator types
abstract class CrossValidator {
  /// Creates k-fold validator to evaluate quality of a predictor.
  ///
  /// It splits a dataset into [numberOfFolds] test sets and subsequently
  /// evaluates given predictor on each produced test set
  ///
  /// Parameters:
  ///
  /// [samples] The whole training dataset to be split into parts to iteratively
  /// evaluate given predictor on the each particular part
  ///
  /// [targetColumnNames] Names of columns from [samples] that contain outcomes
  ///
  /// [numberOfFolds] Number of splits of the [samples]
  ///
  /// [dtype] A type for all the numerical data
  factory CrossValidator.kFold(
      DataFrame samples,
      Iterable<String> targetColumnNames, {
        int numberOfFolds = 5,
        DType dtype = DType.float32,
      }) =>
      CrossValidatorImpl(
        samples,
        targetColumnNames,
        KFoldSplitter(numberOfFolds),
        dtype,
      );

  /// Creates LPO validator to evaluate quality of a predictor.
  ///
  /// It splits a dataset into all possible test sets of size [p] and
  /// subsequently evaluates quality of the predictor on each produced test set.
  ///
  /// Parameters:
  ///
  /// [samples] The whole training dataset to be split into parts to iteratively
  /// evaluate given model on the each particular part.
  ///
  /// [targetColumnNames] Names of columns from [samples] that contain outcomes.
  ///
  /// [p] Size of a split of [samples].
  ///
  /// [dtype] A type for all the numerical data.
  factory CrossValidator.lpo(
      DataFrame samples,
      Iterable<String> targetColumnNames,
      int p, {
        DType dtype = DType.float32,
      }) =>
      CrossValidatorImpl(
        samples,
        targetColumnNames,
        LeavePOutSplitter(p),
        dtype,
      );

  /// Returns a score of quality of passed predictor depending on given
  /// [metricType]
  ///
  /// Parameters:
  ///
  /// [predictorFactory] A factory function that returns a testing predictor
  ///
  /// [metricType] Metric to assess a predictor, that is being created by
  /// [predictorFactory]
  double evaluate(PredictorFactory predictorFactory, MetricType metricType, {
    DataPreprocessFn dataPreprocessFn,
  });
}
