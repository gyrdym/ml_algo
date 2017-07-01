import 'package:dart_vector/vector.dart' show Vector;
import 'package:dart_ml/src/estimator/estimator.dart';

abstract class Predictor {
  Estimator get estimator;
  void train(List<Vector> features, Vector labels, {Vector weights});
  double test(List<Vector> features, Vector origLabels, {Estimator estimator});
  Vector predict(List<Vector> features);
}
