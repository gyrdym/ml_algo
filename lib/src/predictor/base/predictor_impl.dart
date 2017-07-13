import 'dart:typed_data' show Float32List;
import 'package:simd_vector/vector.dart';
import 'package:dart_ml/src/predictor/base/predictor.dart';
import 'package:dart_ml/src/optimizer/interface/optimizer.dart';
import 'package:dart_ml/src/metric/metric.dart';
import 'package:dart_ml/src/score_function/score_function.dart';

abstract class PredictorImpl implements Predictor {
  final Metric metric;
  final ScoreFunction scoreFunction;
  final Optimizer _optimizer;

  Float32x4Vector _weights;

  PredictorImpl(this._optimizer, {Metric metric, ScoreFunction scoreFn}) :
        metric = metric,
        scoreFunction = scoreFn;

  void train(List<Float32x4Vector> features, List<double> labels, {Float32x4Vector weights}) {
    Float32List typedLabelList = new Float32List.fromList(labels);
    _weights = _optimizer.optimize(features, typedLabelList, weights: weights);
  }

  double test(List<Float32x4Vector> features, List<double> origLabels, {Metric metric}) {
    metric = metric ?? this.metric;
    Float32x4Vector prediction = predict(features);
    return metric.getError(prediction, new Float32x4Vector.from(origLabels));
  }

  Float32x4Vector predict(List<Float32x4Vector> features) {
    List<double> labels = new List<double>(features.length);
    for (int i = 0; i < features.length; i++) {
      labels[i] = scoreFunction.score(_weights, features[i]);
    }
    return new Float32x4Vector.from(labels);
  }
}