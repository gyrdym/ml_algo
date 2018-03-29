part of 'package:dart_ml/src/core/implementation.dart';

class _PredictorBase implements Predictor {
  @override
  final Metric metric = coreInjector.get(Metric);
  final ScoreFunction scoreFunction = coreInjector.get(ScoreFunction);
  final Optimizer _optimizer = coreInjector.get(Optimizer);

  Float32x4Vector _weights;

  @override
  void train(
    List<Float32x4Vector> features,
    List<double> labels,
    {Float32x4Vector weights}
  ) {
    final typedLabelList = new Float32List.fromList(labels);
    _weights = _optimizer.findExtrema(features, typedLabelList, weights: weights, isMinimizingObjective: true);
  }

  @override
  double test(
    List<Float32x4Vector> features,
    List<double> origLabels,
    {MetricType metric}
  ) {
    Metric _metric = metric == null ? metric : MetricFactory.createByType(metric);
    Float32x4Vector prediction = predict(features);
    return _metric.getError(prediction, new Float32x4Vector.from(origLabels));
  }

  @override
  Float32x4Vector predict(List<Float32x4Vector> features) {
    final labels = new List<double>(features.length);
    for (int i = 0; i < features.length; i++) {
      labels[i] = scoreFunction.score(_weights, features[i]);
    }
    return new Float32x4Vector.from(labels);
  }
}