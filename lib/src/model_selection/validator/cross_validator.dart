import 'package:dart_vector/vector.dart' show Vector;
import 'package:dart_ml/src/predictor/predictor.dart' show Predictor;
import 'package:dart_ml/src/estimator/estimator.dart';

abstract class ICrossValidator {
  Vector validate(Predictor predictor, List<Vector> features, Vector labels, {Estimator estimator});
}