part of 'package:dart_ml/src/predictor/implementation.dart';

class _ClassifierBase {
  final _PredictorBase _basePredictor = new _PredictorBase();

  Metric get metric => _basePredictor.metric;

  void train(List<Float32x4Vector> features, List<double> labels, {Float32x4Vector weights}) {
    _basePredictor.train(features, labels, weights: weights);
  }

  double test(List<Float32x4Vector> features, List<double> origLabels, {Metric metric}) {
    metric = metric ?? _basePredictor.metric;
    Float32x4Vector prediction = predictClasses(features);
    return metric.getError(prediction, new Float32x4Vector.from(origLabels));
  }

  Float32x4Vector predict(List<Float32x4Vector> features) => _basePredictor.predict(features);

  Float32x4Vector predictClasses(List<Float32x4Vector> features) {
    Float32List probabilities = _basePredictor.predict(features).asList();
    return new Float32x4Vector.from(probabilities.map((double value) => value.round() * 1.0));
  }
}
