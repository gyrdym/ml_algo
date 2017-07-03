import 'package:dart_vector/vector.dart';
import 'package:dart_ml/src/estimator/estimator.dart';

abstract class Predictor {
  Estimator get estimator;
  void train(List<Float32x4Vector> features, List<double> labels, {Float32x4Vector weights});
  double test(List<Float32x4Vector> features, List<double> origLabels, {Estimator estimator});
  Float32x4Vector predict(List<Float32x4Vector> features);
}
