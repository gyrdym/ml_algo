import 'package:dart_ml/src/math/vector/vector.dart';
import 'package:dart_ml/src/estimator/estimator.dart';

abstract class Predictor {
  Estimator get estimator;
  void train(List<Vector> features, Vector labels, {Vector weights});
  double test(List<Vector> features, Vector origLabels, {Estimator estimator});
  Vector predict(List<Vector> features);
}
