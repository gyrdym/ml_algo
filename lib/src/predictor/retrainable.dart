import 'package:ml_algo/src/predictor/predictor.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

abstract class Retrainable {
  /// Re-runs the process on new training [data]. The features, model algorithm,
  /// and hyperparameters remain the same.
  Predictor retrain(DataFrame data);
}
