import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';

/// A common interface for all types of classifiers and regressors
abstract class Predictor {
  /// A collection of target column names of a dataset used to learn the ML
  /// model
  Iterable<String> get targetNames;

  /// A type for all the numeric values using by the [Predictor]
  DType get dtype;

  /// Assesses model performance according to provided [metricType]
  ///
  /// Throws an exception if inappropriate [metricType] provided.
  double assess(DataFrame observations, MetricType metricType);

  /// Returns prediction, based on the model learned parameters
  DataFrame predict(DataFrame testFeatures);

  /// Re-runs the process on new training [data]. The features, model algorithm,
  /// and hyperparameters remain the same.
  Predictor retrain(DataFrame data);
}
