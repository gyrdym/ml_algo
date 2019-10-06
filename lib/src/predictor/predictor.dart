import 'package:ml_dataframe/ml_dataframe.dart';

/// A common interface for all types of classifiers and regressors
abstract class Predictor {
  /// Returns prediction, based on the model learned parameters
  DataFrame predict(DataFrame features);
}
