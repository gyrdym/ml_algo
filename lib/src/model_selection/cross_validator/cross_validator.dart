import 'package:ml_algo/src/common/constants/default_parameters/common.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/model_selection/_init_module.dart';
import 'package:ml_algo/src/model_selection/_injector.dart';
import 'package:ml_algo/src/model_selection/assessable.dart';
import 'package:ml_algo/src/model_selection/cross_validator/cross_validator_impl.dart';
import 'package:ml_algo/src/model_selection/split_indices_provider/split_indices_provider_factory.dart';
import 'package:ml_algo/src/model_selection/split_indices_provider/split_indices_provider_type.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/vector.dart';

typedef ModelFactory = Assessable Function(DataFrame observations);

typedef DataPreprocessFn = List<DataFrame> Function(
    DataFrame trainData, DataFrame testData);

/// A factory and an interface for all the cross validator types
abstract class CrossValidator {
  /// Creates a k-fold validator to evaluate the quality of a ML model.
  ///
  /// It splits a dataset into [numberOfFolds] test sets and evaluates a model
  /// on each produced test set
  ///
  /// Parameters:
  ///
  /// [samples] A dataset that is going to be split into [numberOfFolds] parts
  /// to iteratively evaluate on them a model's performance
  ///
  /// [numberOfFolds] A number of parts of the [samples]
  ///
  /// [dtype] A type for all numerical data
  factory CrossValidator.kFold(
    DataFrame samples, {
    int numberOfFolds = 5,
    DType dtype = dTypeDefaultValue,
  }) {
    initModelSelectionModule();

    final dataSplitterFactory =
        modelSelectionInjector.get<SplitIndicesProviderFactory>();
    final dataSplitter = dataSplitterFactory.createByType(
        SplitIndicesProviderType.kFold,
        numberOfFolds: numberOfFolds);

    return CrossValidatorImpl(
      samples,
      dataSplitter,
      dtype,
    );
  }

  /// Creates a LPO validator to evaluate the quality of a ML model.
  ///
  /// It splits a dataset into all possible test sets of size [p] and evaluates
  /// the quality of a model on each produced test set.
  ///
  /// Parameters:
  ///
  /// [samples] A dataset that is going to be split into parts to iteratively
  /// evaluate on them a model's performance
  ///
  /// [p] A size of a part of [samples] in rows.
  ///
  /// [dtype] A type for all the numerical data.
  factory CrossValidator.lpo(
    DataFrame samples,
    int p, {
    DType dtype = dTypeDefaultValue,
  }) {
    initModelSelectionModule();

    final dataSplitterFactory =
        modelSelectionInjector.get<SplitIndicesProviderFactory>();
    final dataSplitter =
        dataSplitterFactory.createByType(SplitIndicesProviderType.lpo, p: p);

    return CrossValidatorImpl(
      samples,
      dataSplitter,
      dtype,
    );
  }

  /// Returns a future that is resolved with a vector of scores of quality of a
  /// model depending on given [metricType]
  ///
  /// Parameters:
  ///
  /// [createModel] A function that returns a model to be evaluated
  ///
  /// [metricType] A metric used to assess a model created by [createModel]
  ///
  /// [onDataSplit] A callback that is called when a new train-test split is
  /// ready to be passed into a model. One may place some additional
  /// data-dependent logic here, e.g., data preprocessing. The callback accepts
  /// train and test data from a new split and returns a transformed split as a
  /// list, where the first element is train data and the second one is test
  /// data, both of [DataFrame] type. This new transformed split will be passed
  /// into the model
  ///
  /// Example:
  ///
  /// ````dart
  /// final data = DataFrame([
  ///     [ 1,  1,  1,   1],
  ///     [ 2,  3,  4,   5],
  ///     [18, 71, 15,  61],
  ///     [19,  0, 21, 331],
  ///     [11, 10,  9,  40],
  ///   ],
  ///   header: header,
  ///   headerExists: false,
  /// );
  /// final modelFactory = (trainData) =>
  ///   KnnRegressor(trainData, 'col_3', k: 4);
  /// final onDataSplit = (trainData, testData) {
  ///   final standardizer = Standardizer(trainData);
  ///   return [
  ///     standardizer.process(trainData),
  ///     standardizer.process(testData),
  ///   ];
  /// }
  /// final validator = CrossValidator.kFold(data);
  /// final scores = await validator.evaluate(
  ///   modelFactory,
  ///   MetricType.mape,
  ///   onDataSplit: onDataSplit,
  /// );
  /// final averageScore = scores.mean();
  ///
  /// print(averageScore);
  /// ````
  Future<Vector> evaluate(
    ModelFactory createModel,
    MetricType metricType, {
    DataPreprocessFn? onDataSplit,
  });
}
