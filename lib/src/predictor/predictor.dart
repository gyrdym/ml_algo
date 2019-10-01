import 'package:ml_dataframe/ml_dataframe.dart';

abstract class Predictor {
  /// Returns prediction, based on the model learned parameters
  DataFrame predict(DataFrame features);
}
