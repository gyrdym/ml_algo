import 'package:simd_vector/vector.dart';
import 'package:dart_ml/src/metric/metric.dart';
import 'package:dart_ml/src/optimizer/gradient/interface/stochastic.dart';
import 'package:dart_ml/src/predictor/linear/base/gradient_predictor.dart';
import 'classifier.dart';

class GradientLinearClassifier extends GradientLinearPredictor implements Classifier {
  GradientLinearClassifier(SGDOptimizer optimizer, Metric metric)
      : super(optimizer, metric: metric);

  Float32x4Vector predictProbabilities(List<Float32x4Vector> features) =>
      new Float32x4Vector.from(predict(features).asList().map((double value) => value > 0.5 ? 1.0 : 0.0));
}
