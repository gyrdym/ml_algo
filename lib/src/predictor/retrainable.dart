import 'package:ml_algo/src/predictor/predictor.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

abstract class Retrainable<T extends Predictor> {
  /// Re-runs the learning process on the new training [data]. The features, model algorithm,
  /// and hyperparameters remain the same.
  T retrain(DataFrame data);
}
