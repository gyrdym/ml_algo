import 'package:dart_ml/src/utils/generic_type_instantiator.dart';
import 'package:dart_ml/src/math/vector_interface.dart';
import 'package:dart_ml/src/predictors/predictor_interface.dart';
import 'package:dart_ml/src/optimizers/gradient_optimizer.dart';

class GradientLinearRegressor<T extends VectorInterface, O extends GradientOptimizer<T>> implements PredictorInterface<T> {
  O _optimizer;
  T _weights;

  T get weights => _weights;

  GradientLinearRegressor({double learningRate = 1e-5, double minWeightsDistance = 1e-8, int iterationLimit = 10000})
      : _optimizer = Instantiator.createInstance(O, const Symbol(''), [learningRate, minWeightsDistance, iterationLimit]);

  void train(List<T> features, List<double> labels) {
    _weights = _optimizer.optimize(features, labels);
  }

  T predict(List<T> features) {
    List<double> labels = new List(features.length);
    for (int i = 0; i < features.length; i++) {
      labels[i] = _weights.vectorScalarMult(features[i]);
    }
    return Instantiator.createInstance(T, const Symbol('from'), [labels]);
  }
}
