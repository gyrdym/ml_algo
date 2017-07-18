import 'package:dart_ml/src/di/injector.dart';
import 'package:dart_ml/src/metric/metric.dart';
import 'package:dart_ml/src/optimizer/gradient/stochastic.dart';
import 'package:dart_ml/src/optimizer/regularization.dart';
import 'package:dart_ml/src/loss_function/loss_function.dart';
import 'package:dart_ml/src/score_function/score_function.dart';
import 'package:dart_ml/src/predictor/linear/base/gradient_predictor.dart';

class SGDRegressor extends GradientLinearPredictorBase {
  SGDRegressor({double learningRate,
                 double minWeightsDistance,
                 int iterationLimit,
                 Metric metric,
                 Regularization regularization,
                 alpha})
      : super(
      (injector.get(SGDOptimizer) as SGDOptimizer)
        ..configure(
            learningRate: learningRate,
            minWeightsDistance: minWeightsDistance,
            iterationLimit: iterationLimit,
            regularization: regularization,
            lossFunction: new LossFunction.Squared(),
            scoreFunction: new ScoreFunction.Linear(),
            alpha: alpha),

      metric: metric);
}
