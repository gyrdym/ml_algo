import 'dart:math' as math;
import 'package:dart_ml/src/math/vector_interface.dart';
import 'package:dart_ml/src/predictors/predictor.dart';
import 'package:dart_ml/src/enums.dart';

class SGDLinearRegressor implements Predictor {
  double step;
  double minWeightsDistance;
  int iterationLimit;
  List<double> errors;

  VectorInterface _weights;
  double _rmse;

  VectorInterface get weights => _weights;
  double get rmse => _rmse;

  SGDLinearRegressor({this.step = 1e-8, this.minWeightsDistance = 1e-8, this.iterationLimit = 1000});

  void train(List<VectorInterface> features, VectorInterface labels, [CostFunction metric = CostFunction.RMSE]) {
    _addBiasTo(features);

    _weights = _optimize(features, labels, metric);
    _rmse = _calculateRMSE(features, labels);
  }

  VectorInterface predict(List<VectorInterface> features) {
    VectorInterface labels = new Object() as VectorInterface;

    for (int i = 0; i < features.length; i++) {
      labels[i] = _weights.vectorScalarMult(features[i]);
    }

    return labels;
  }

  void _addBiasTo(List<VectorInterface> features) {
    for (int i = 0; i < features.length; i++) {
      features[i].add(1.0);
    }
  }

  List<double> _optimize(List<VectorInterface> features, VectorInterface labels, CostFunction metric) {
    math.Random randomizer = new math.Random();
    double weightsDistance = double.MAX_FINITE;
    int iterationCounter = 0;

    VectorInterface weights = new Object() as VectorInterface;

    //vectors.create(features.first.length, 0.0);

    while (weightsDistance > minWeightsDistance && iterationCounter < iterationLimit) {
      int k = randomizer.nextInt(features.length);

      double eta = step / (iterationCounter + 1);
      VectorInterface newWeights = _doIteration(weights, features[k], labels[k], eta);
      weightsDistance = newWeights.distanceTo(weights);
      weights = newWeights;

      iterationCounter++;
    }

    return weights;
  }

  VectorInterface _doIteration(VectorInterface weights, VectorInterface features, double y, double eta) {
    VectorInterface newWeights = new Object() as VectorInterface;
    double delta = weights.vectorScalarMult(features) - y;

    for (int i = 0; i < features.dimension; i++) {
      newWeights.add(weights[i] - (2 * eta) * delta * features[i]);
    }

    return newWeights;
  }

  double _calculateRMSE(List<VectorInterface> features, List<double> labels) {
    List<double> errors = [];

    for (int i = 0; i < features.length; i++) {
      double predictedLabel = _weights.vectorScalarMult(features[i]);
      double label = labels[i];
      double delta = predictedLabel - label;
      errors.add(math.pow(delta, 2));
    }

    return math.sqrt(errors.reduce((double item, double sum) => sum += item) / features.length);
  }
}
