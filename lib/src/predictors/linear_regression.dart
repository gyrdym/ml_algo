import 'dart:math' as math;

import 'package:dart_ml/src/vector_operations.dart' as vectors;
import 'package:dart_ml/src/optimizers/optimizer.dart';
import 'package:dart_ml/src/optimizers/sgd.dart';
import 'package:dart_ml/src/predictors/predictor.dart';
import 'package:dart_ml/src/enums.dart';

class LinearRegression implements Predictor {
  List<double> _weights;
  Optimizer _optimizer;

  void train(List<List<num>> features, List<num> labels,
      [OptimizationMethod opMethod = OptimizationMethod.SGD, CostFunction metric = CostFunction.RMSE]) {
    _weights = new List<double>();

    int dimension = features.first.length;
    List<num> biasFeatures = vectors.create(dimension, 1.0);

    features.add(biasFeatures);

    _weights = _calculateWeights(features, labels, opMethod, metric);
  }

  List<num> predict(List<List<num>> features) {
    List<num> labels = new List<num>();

    for (int i = 0; i < features.length; i++) {
      labels.add(vectors.scalarMult(_weights, features[i]));
    }

    return labels;
  }

  List<double> get weights => _weights;
  Optimizer get optimizer => _optimizer;

  List<double> _calculateWeights(List<List<num>> features, List<num> labels,
      OptimizationMethod method, CostFunction metric) {
    switch (method) {
      case OptimizationMethod.SGD:
        _optimizer = new StochasticGradientDescent();
    }

    return _optimizer.optimize(features, labels);
  }

  double _calculateMSE(num label, num predictedLabel) =>
      math.pow((label - predictedLabel), 2);
}
