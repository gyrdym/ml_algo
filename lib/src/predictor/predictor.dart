import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';

/// A common interface for all types of classifiers and regressors
abstract class Predictor {
  /// A collection of target column names of a dataset which was used to learn the ML
  /// model
  Iterable<String> get targetNames;

  /// Returns prediction, based on the learned coefficients
  DataFrame predict(DataFrame testFeatures);

  /// A type for all numeric values using by the [Predictor]
  DType get dtype;
}
