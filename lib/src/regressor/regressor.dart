import 'package:ml_algo/src/predictor/predictor.dart';
import 'package:ml_linalg/matrix.dart';

abstract class Regressor implements Predictor {
  Matrix predict(Matrix features);
}
