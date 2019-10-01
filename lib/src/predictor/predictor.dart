import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/matrix.dart';

abstract class Predictor {
  /// Returns prediction, based on the model learned parameters
  DataFrame predict(Matrix features);
}
