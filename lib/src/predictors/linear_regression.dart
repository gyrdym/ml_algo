import 'package:dart_ml/src/vector_operations.dart' as vectors;
import 'package:dart_ml/src/optimizers/optimizer.dart';
import 'package:dart_ml/src/optimizers/sgd.dart';
import 'package:dart_ml/src/predictors/predictor.dart';
import 'package:dart_ml/src/enums.dart';

class LinearRegression implements Predictor {
  List<double> _weights;
  Optimizer _optimizer;

  List<double> get weights => _weights;
  Optimizer get optimizer => _optimizer;

  void train(List<List<double>> features, List<double> labels,
      [OptimizationMethod opMethod = OptimizationMethod.SGD, CostFunction metric = CostFunction.RMSE]) {
    _weights = new List<double>();

    int dimension = features.first.length;
    List<double> biasFeatures = vectors.create(dimension, 1.0);

    features.add(biasFeatures);

    _weights = _calculateWeights(features, labels, opMethod, metric);
  }

  List<double> predict(List<List<double>> features) {
    List<double> labels = new List<double>();

    for (int i = 0; i < features.length; i++) {
      labels.add(vectors.scalarMult(_weights, features[i]));
    }

    return labels;
  }

  List<double> _calculateWeights(List<List<double>> features, List<double> labels,
      OptimizationMethod method, CostFunction metric) {
    switch (method) {
      case OptimizationMethod.SGD:
        _optimizer = new StochasticGradientDescent();
    }

    return _optimizer.optimize(features, labels);
  }
}
