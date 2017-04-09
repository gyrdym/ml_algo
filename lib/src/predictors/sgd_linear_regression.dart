import 'dart:math' as math;
import 'package:dart_ml/src/math/typed_vector.dart' as vectors;
import 'package:dart_ml/src/predictors/predictor.dart';
import 'package:dart_ml/src/enums.dart';

class SGDLinearRegressor implements Predictor {
  double step;
  double minWeightsDistance;
  int iterationLimit;
  List<double> errors;

  List<double> _weights;
  double _rmse;

  List<double> get weights => _weights;
  double get rmse => _rmse;

  SGDLinearRegressor({this.step = 1e-8, this.minWeightsDistance = 1e-8, this.iterationLimit = 1000});

  void train(List<List<double>> features, List<double> labels, [CostFunction metric = CostFunction.RMSE]) {
    _addBias(features);

    _weights = _optimize(features, labels, metric);
    _rmse = _calculateRMSE(features, labels);
  }

  List<double> predict(List<List<double>> features) {
    List<double> labels = new List<double>();

    for (int i = 0; i < features.length; i++) {
      labels.add(vectors.scalarMult(_weights, features[i]));
    }

    return labels;
  }

  void _addBias(List<List<double>> features) {
    for (int i = 0; i < features.length; i++) {
      features[i].add(1.0);
    }
  }

  List<double> _optimize(List<List<double>> features, List<double> labels, CostFunction metric) {
    math.Random randomizer = new math.Random();
    double weightsDistance = double.MAX_FINITE;
    int iterationCounter = 0;

    List<double> weights = vectors.create(features.first.length, 0.0);

    while (weightsDistance > minWeightsDistance && iterationCounter < iterationLimit) {
      int k = randomizer.nextInt(features.length);

      double eta = step / (iterationCounter + 1);
      List<double> newWeights = _doIteration(weights, features[k], labels[k], eta);
      weightsDistance = vectors.distance(newWeights, weights);
      weights = newWeights;

      iterationCounter++;
    }

    return weights;
  }

  List<double> _doIteration(List<double> weights, List<double> features, double y, double eta) {
    int dimensions = features.length;
    List<double> newWeights = new List<double>();
    double delta = vectors.scalarMult(weights, features) - y;

    for (int i = 0; i < dimensions; i++) {
      double w = weights[i];
      double x = features[i];

      newWeights.add(w - (2 * eta) * delta * x);
    }

    return newWeights;
  }

  double _calculateRMSE(List<List<double>> features, List<double> labels) {
    List<double> errors = [];

    for (int i = 0; i < features.length; i++) {
      double predictedLabel = vectors.scalarMult(_weights, features[i]);
      double label = labels[i];
      double delta = predictedLabel - label;
      errors.add(math.pow(delta, 2));
    }

    return math.sqrt(errors.reduce((double item, double sum) => sum += item) / features.length);
  }
}
