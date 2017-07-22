import 'package:di/di.dart';
import 'dart:typed_data' show Float32List;
import 'package:simd_vector/vector.dart';
import 'package:dart_ml/src/predictor/base/predictor.dart';
import 'package:dart_ml/src/optimizer/base.dart';
import 'package:dart_ml/src/optimizer/gradient/base.dart';
import 'package:dart_ml/src/metric/metric.dart';
import 'package:dart_ml/src/score_function/score_function.dart';
import 'package:dart_ml/src/predictor/base/classifier.dart';
import 'package:dart_ml/src/di/injector.dart';
import 'package:dart_ml/src/di/dependencies.dart';
import 'package:dart_ml/src/optimizer/regularization.dart';
import 'package:dart_ml/src/loss_function/loss_function.dart';
import 'package:dart_ml/src/optimizer/gradient/batch.dart';
import 'package:dart_ml/src/optimizer/gradient/mini_batch.dart';
import 'package:dart_ml/src/optimizer/gradient/stochastic.dart';

part 'linear/base/gradient_predictor.dart';
part 'linear/classifier/gradient/gradient_classifier.dart';
part 'linear/classifier/gradient/logistic_regression.dart';
part 'linear/regressor/gradient/regressor.dart';
part 'linear/regressor/gradient/batch.dart';
part 'linear/regressor/gradient/mini_batch.dart';
part 'linear/regressor/gradient/stochastic.dart';

abstract class PredictorBase implements Predictor {
  final Metric metric;
  final ScoreFunction scoreFunction;

  Optimizer _optimizer;
  Float32x4Vector _weights;

  PredictorBase({Metric metric, ScoreFunction scoreFn}) :
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