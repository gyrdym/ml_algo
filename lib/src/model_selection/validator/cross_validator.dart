import 'package:dart_vector/vector.dart';
import 'package:dart_ml/src/predictor/predictor.dart' show Predictor;
import 'package:dart_ml/src/estimator/estimator.dart';

abstract class ICrossValidator {
  Float32x4Vector validate(Predictor predictor, List<Float32x4Vector> features, List<double> labels, {Estimator estimator});
}