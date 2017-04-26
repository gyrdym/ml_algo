import 'package:dart_ml/src/math/vector/vector_factory.dart';
import 'package:dart_ml/src/math/vector/vector_interface.dart';
import 'package:dart_ml/src/predictors/predictor_interface.dart';
import 'package:dart_ml/src/optimizers/gradient/base_optimizer.dart';

abstract class LinearRegressor<T extends VectorInterface> implements PredictorInterface<T> {
  GradientOptimizer<T> optimizer;
  T _weights;

  T get weights => _weights;

  void train(List<T> features, List<double> labels) {
    _weights = optimizer.optimize(features, labels);
  }

  T predict(List<T> features) {
    List<double> labels = new List(features.length);
    for (int i = 0; i < features.length; i++) {
      labels[i] = _weights.vectorScalarMult(features[i]);
    }
    return VectorFactory.createFrom(T, labels);
  }
}
