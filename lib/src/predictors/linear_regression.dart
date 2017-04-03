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

  LinearRegression([OptimizationMethod opMethod = OptimizationMethod.SGD]) {
    switch (opMethod) {
      case OptimizationMethod.SGD:
        _optimizer = new StochasticGradientDescent();
    }
  }

  void train(List<List<double>> features, List<double> labels,
      [OptimizationMethod opMethod = OptimizationMethod.SGD, CostFunction metric = CostFunction.RMSE]) {
    _weights = new List<double>();

    _addBias(features);

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
    return _optimizer.optimize(features, labels);
  }

  void _addBias(List<List<double>> features) {
    for (int i = 0; i < features.length; i++) {
      features[i].add(1.0);
    }
  }
}
