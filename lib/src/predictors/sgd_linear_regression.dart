import 'dart:mirrors';
import 'dart:math' as math;
import 'package:dart_ml/src/math/vector_interface.dart';
import 'package:dart_ml/src/math/regular_vector.dart';
import 'package:dart_ml/src/predictors/predictor.dart';
import 'package:dart_ml/src/enums.dart';

class SGDLinearRegressor<T extends VectorInterface> implements Predictor<T> {
  double step;
  double minWeightsDistance;
  int iterationLimit;
  List<double> errors;

  T _weights;
  double _rmse;

  T get weights => _weights;
  double get rmse => _rmse;

  SGDLinearRegressor({this.step = 1e-8, this.minWeightsDistance = 1e-8, this.iterationLimit = 1000});

  void train(List<T> features, T labels, [CostFunction metric = CostFunction.RMSE]) {
    _addBiasTo(features);
    _weights = _optimize(features, labels, metric);
    _rmse = _calculateRMSE(features, labels);
  }

  T predict(List<T> features) {
    List<double> labels = new List(_weights.dimension);

    for (int i = 0; i < features.length; i++) {
      labels[i] = _weights.vectorScalarMult(features[i]);
    }

    return (reflectType(T) as ClassMirror).newInstance(const Symbol('fromList'), [labels]).reflectee;
  }

  void _addBiasTo(List<T> features) {
    features.forEach((T vector) => vector.add(1.0));
  }

  T _optimize(List<T> features, T labels, CostFunction metric) {
    math.Random randomizer = new math.Random();
    double weightsDistance = double.MAX_FINITE;
    int iterationCounter = 0;

    T weights = (reflectType(T) as ClassMirror).newInstance(
        const Symbol('filled'), [features.first.dimension, 0.0]).reflectee;

    while (weightsDistance > minWeightsDistance && iterationCounter < iterationLimit) {
      int k = randomizer.nextInt(features.length);
      double eta = step / (iterationCounter + 1);
      T newWeights = _doIteration(weights, features[k], labels[k], eta);
      weightsDistance = newWeights.distanceTo(weights);
      weights = newWeights;
      iterationCounter++;
    }

    return weights;
  }

  T _doIteration(T weights, T features, double y, double eta) {
    var newWeights = new List<double>.filled(features.dimension, 0.0);
    var delta = weights.vectorScalarMult(features) - y;

    for (int i = 0; i < features.dimension; i++) {
      newWeights[i] = weights[i] - (2 * eta) * delta * features[i];
    }

    return (reflectType(T) as ClassMirror).newInstance(const Symbol('from'), [newWeights]).reflectee;
  }

  double _calculateRMSE(List<T> features, T labels) {
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
