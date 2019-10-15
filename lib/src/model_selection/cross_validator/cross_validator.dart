import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/model_selection/assessable.dart';
import 'package:ml_algo/src/model_selection/cross_validator/cross_validator_impl.dart';
import 'package:ml_algo/src/model_selection/data_splitter/data_splitter_factory.dart';
import 'package:ml_algo/src/model_selection/data_splitter/data_splitter_type.dart';
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
      }) {
    final dataSplitterFactory = getDependencies()
        .getDependency<DataSplitterFactory>();
    final dataSplitter = dataSplitterFactory
        .createByType(DataSplitterType.kFold, numberOfFolds: numberOfFolds);

    return CrossValidatorImpl(
      samples,
      targetColumnNames,
      dataSplitter,
      dtype,
    );
  }

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
      }) {
    final dataSplitterFactory = injector.getDependency<DataSplitterFactory>();
    final dataSplitter = dataSplitterFactory
        .createByType(DataSplitterType.lpo, p: p);

    return CrossValidatorImpl(
      samples,
      targetColumnNames,
      dataSplitter,
      dtype,
    );
  }

  /// Returns a score of quality of passed predictor depending on given
  /// [metricType]
  ///
  /// Parameters:
  ///
  /// [predictorFactory] A factory function that returns a testing predictor
  ///
  /// [metricType] Metric to assess a predictor, that is being created by
  /// [predictorFactory]
  ///
  /// [onDataSplit] A callback that is called when a new train-test split is
  /// ready to be passed into evaluating predictor. One may place some
  /// additional data-dependent logic here, e.g., data preprocessing. The
  /// callback accepts train and test data from a new split and returns
  /// transformed split as list, where the first element is training data and
  /// the second one - test data, both of [DataFrame] type. This new transformed
  /// split will be passed into the predictor.
  ///
  /// Example:
  ///
  /// ````dart
  /// final data = DataFrame(
  ///   <Iterable<num>>[
  ///     [ 1,  1,  1,   1],
  ///     [ 2,  3,  4,   5],
  ///     [18, 71, 15,  61],
  ///     [19,  0, 21, 331],
  ///     [11, 10,  9,  40],
  ///   ],
  ///   header: header,
  ///   headerExists: false,
  /// );
  ///
  /// final predictorFactory = (trainData, _) =>
  ///   KnnRegressor(trainData, 'col_3', k: 4);
  ///
  /// final onDataSplit = (trainData, testData) {
  ///   final standardizer = Standardizer(trainData);
  ///   return [
  ///     standardizer.process(trainData),
  ///     standardizer.process(testData),
  ///   ];
  /// }
  ///
  /// final validator = CrossValidator.kFold(data, ['col_3']);
  /// final score = validator.evaluate(
  ///   predictorFactory,
  ///   MetricType.mape,
  ///   onDataSplit: onDataSplit,
  /// );
  /// ````
  double evaluate(PredictorFactory predictorFactory, MetricType metricType, {
    DataPreprocessFn onDataSplit,
  });
}
