library linear_regressor;

import 'package:dart_ml/src/math/vector/vector.dart';
import 'package:dart_ml/src/predictor/predictor.dart';
import 'package:dart_ml/src/predictor/gradient_descent_type.dart';
import 'package:dart_ml/src/predictor/estimator_type.dart';
import 'package:dart_ml/src/optimizer/optimizer.dart';
import 'package:dart_ml/src/optimizer/gradient/batch_optimizer.dart';
import 'package:dart_ml/src/optimizer/gradient/stochastic_optimizer.dart';
import 'package:dart_ml/src/optimizer/gradient/mini_batch_optimizer.dart';
import 'package:dart_ml/src/estimator/estimator.dart';
import 'package:dart_ml/src/estimator/rmse.dart';

part 'optimizer_factory.dart';
part 'estimator_factory.dart';

class LinearRegressor implements Predictor {
  final Estimator estimator;
  final Optimizer _optimizer;

  Vector _weights;

  LinearRegressor({double learningRate = 1e-5, double minWeightsDistance = 1e-8, int iterationLimit = 10000,
                    GradientDescent optimizerType = GradientDescent.MBGD, EstimatorType estimatorType = EstimatorType.RMSE})
      : _optimizer = _OptimizerFactory.create(optimizerType),
        estimator = _EstimatorFactory.create(estimatorType);

  void train(List<Vector> features, Vector labels, {Vector weights}) {
    _weights = _optimizer.optimize(features, labels, weights: weights);
  }

  double test(List<Vector> features, Vector origLabels, {Estimator estimator}) {
    estimator = estimator ?? this.estimator;
    Vector prediction = predict(features);
    return estimator.calculateError(prediction, origLabels);
  }

  Vector predict(List<Vector> features) {
    List<double> labels = new List<double>(features.length);
    for (int i = 0; i < features.length; i++) {
      labels[i] = _weights.dot(features[i]);
    }
    return new Vector.from(labels);
  }
}
