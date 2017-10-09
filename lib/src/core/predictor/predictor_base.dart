part of 'package:dart_ml/src/core/implementation.dart';

class _PredictorBase {
  final Metric metric = coreInjector.get(Metric);
  final ScoreFunction scoreFunction = coreInjector.get(ScoreFunction);
  final Optimizer _optimizer = coreInjector.get(Optimizer);

  Float32x4Vector _weights;

  void train(List<Float32x4Vector> features, List<double> labels, {Float32x4Vector weights}) {
    Float32List typedLabelList = new Float32List.fromList(labels);
    _weights = _optimizer.findMinima(features, typedLabelList, weights: weights);
  }

  double test(List<Float32x4Vector> features, List<double> origLabels, {MetricType metricType}) {
    Metric _metric = metricType == null ? metric : MetricFactory.createByType(metricType);
    Float32x4Vector prediction = predict(features);
    return _metric.getError(prediction, new Float32x4Vector.from(origLabels));
  }

  Float32x4Vector predict(List<Float32x4Vector> features) {
    List<double> labels = new List<double>(features.length);
    for (int i = 0; i < features.length; i++) {
      labels[i] = scoreFunction.score(_weights, features[i]);
    }
    return new Float32x4Vector.from(labels);
  }
}