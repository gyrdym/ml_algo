import 'dart:typed_data' show Float32List;
import 'package:simd_vector/vector.dart';
import 'package:dart_ml/src/metric/metric.dart';
import 'package:dart_ml/src/optimizer/gradient/stochastic.dart';
import 'package:dart_ml/src/predictor/linear/base/gradient_predictor.dart';
import 'package:dart_ml/src/predictor/base/classifier.dart';

class GradientLinearClassifier extends GradientLinearPredictorBase implements Classifier {
  GradientLinearClassifier(SGDOptimizer optimizer, Metric metric)
      : super(optimizer, metric: metric);

  @override
  double test(List<Float32x4Vector> features, List<double> origLabels, {Metric metric}) {
    metric = metric ?? this.metric;
    Float32x4Vector prediction = predictClasses(features);
    return metric.getError(prediction, new Float32x4Vector.from(origLabels));
  }

  Float32x4Vector predictClasses(List<Float32x4Vector> features) {
    Float32List probabilities = predict(features).asList();
    return new Float32x4Vector.from(probabilities.map((double value) => value.round() * 1.0));
  }
}
