part of '../../../predictor.dart';

class GradientLinearClassifier extends GradientLinearPredictor implements Classifier {
  GradientLinearClassifier(Metric metric) : super(metric: metric);

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
