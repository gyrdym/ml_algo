import 'package:dart_ml/src/math/vector/vector.dart';
import 'package:dart_ml/src/predictor/predictor.dart' show Predictor;
import 'package:dart_ml/src/estimator/estimator.dart';

abstract class ICrossValidator {
  Vector validate(Predictor predictor, List<Vector> features, Vector labels, {Estimator estimator});
}