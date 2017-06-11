import 'package:dart_ml/src/math/vector/vector.dart';
import 'package:dart_ml/src/predictor/interface/predictor.dart';
import 'package:dart_ml/src/estimator/estimator.dart';

abstract class CrossValidator {
  Vector validate(Predictor predictor, List<Vector> features, Vector labels, {Estimator estimator});
}