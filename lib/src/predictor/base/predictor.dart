import 'package:simd_vector/vector.dart';
import 'package:dart_ml/src/metric/metric.dart';
import 'package:dart_ml/src/score_function/score_function.dart';

abstract class Predictor {
  Metric get metric;
  ScoreFunction get scoreFunction;
  void train(List<Float32x4Vector> features, List<double> labels, {Float32x4Vector weights});
  double test(List<Float32x4Vector> features, List<double> origLabels, {Metric metric});
  Float32x4Vector predict(List<Float32x4Vector> features);
}
