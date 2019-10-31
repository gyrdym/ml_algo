import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';

/// A common interface for all types of classifiers and regressors
abstract class Predictor {
  /// Returns prediction, based on the model learned parameters
  DataFrame predict(DataFrame testFeatures);

  DType get dtype;
}
