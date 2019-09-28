import 'package:ml_linalg/matrix.dart';

abstract class Predictor {
  /// Returns prediction, based on the model learned parameters
  Matrix predict(Matrix features);
}
