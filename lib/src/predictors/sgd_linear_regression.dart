import 'dart:mirrors';
import 'package:dart_ml/src/math/vector_interface.dart';
import 'package:dart_ml/src/predictors/predictor_interface.dart';
import 'package:dart_ml/src/optimizers/sgd_optimizer.dart';
import 'package:dart_ml/src/enums.dart';

class SGDLinearRegressor<T extends VectorInterface> implements PredictorInterface<T> {
  SGDOptimizer<T> _optimizer;
  T _weights;

  T get weights => _weights;
  SGDOptimizer<T> get optimizer => _optimizer;

  SGDLinearRegressor({double step = 1e-5, double minWeightsDistance = 1e-8, int iterationLimit = 10000})
      : _optimizer = new SGDOptimizer<T>(step: step, minWeightsDistance: minWeightsDistance, iterationLimit: iterationLimit);

  void train(List<T> features, T labels, [CostFunction metric = CostFunction.RMSE]) {
    _addBiasTo(features);
    _weights = optimizer.optimize(features, labels, metric);
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
}
