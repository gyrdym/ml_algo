import 'dart:mirrors';
import 'dart:math' as math;
import 'package:dart_ml/src/math/vector_interface.dart';
import 'package:dart_ml/src/predictors/predictor.dart';
import 'package:dart_ml/src/estimators/rmse.dart';
import 'package:dart_ml/src/enums.dart';

class SGDLinearRegressor<T extends VectorInterface> implements Predictor<T> {
  final RMSEEstimator _estimator = new RMSEEstimator();

  double step;
  double minWeightsDistance;
  int iterationLimit;
  List<double> errors;

  T _weights;

  T get weights => _weights;
  RMSEEstimator get estimator => _estimator;

  SGDLinearRegressor({this.step = 1e-8, this.minWeightsDistance = 1e-8, this.iterationLimit = 10000});

  void train(List<T> features, T labels, [CostFunction metric = CostFunction.RMSE]) {
    _addBiasTo(features);
    _weights = _optimize(features, labels, metric);
  }

  T predict(List<T> features) {
    _addBiasTo(features);
    List<double> labels = new List(features.length);
    for (int i = 0; i < features.length; i++) {
      labels[i] = _weights.vectorScalarMult(features[i]);
    }
    return (reflectType(T) as ClassMirror).newInstance(const Symbol('from'), [labels]).reflectee;
  }

  void _addBiasTo(List<T> features) {
    features.forEach((T vector) => vector.add(1.0));
  }

  T _optimize(List<T> features, T labels, CostFunction metric) {
    math.Random randomizer = new math.Random();
    double weightsDistance = double.MAX_FINITE;
    int iterationCounter = 0;

    T weights = (reflectType(T) as ClassMirror).newInstance(
        const Symbol('filled'), [features.first.length, 0.0]).reflectee;

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
    var newWeights = new List<double>.filled(features.length, 0.0);
    var delta = weights.vectorScalarMult(features) - y;

    for (int i = 0; i < features.length; i++) {
      newWeights[i] = weights[i] - (2 * eta) * delta * features[i];
    }

    return (reflectType(T) as ClassMirror).newInstance(const Symbol('from'), [newWeights]).reflectee;
  }
}
