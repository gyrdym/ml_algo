import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer.dart';
import 'package:ml_linalg/matrix.dart';

class NewtonOptimizer implements LinearOptimizer {
  NewtonOptimizer(
      this._features, this._labels, this._costFunction, this._iterationLimit);

  final Matrix _features;
  final Matrix _labels;
  final CostFunction _costFunction;
  final int _iterationLimit;
  final List<num> _costPerIteration = [];

  @override
  List<num> get costPerIteration => _costPerIteration;

  @override
  Matrix findExtrema(
      {Matrix? initialCoefficients,
      bool isMinimizingObjective = true,
      bool collectLearningData = false}) {
    var dtype = _features.dtype;
    var i = 0;
    var coefficients =
        Matrix.column(List.filled(_features.first.length, 0), dtype: dtype);
    final inverseHessian =
        _costFunction.getHessian(_features, coefficients, _labels).inverse();

    while (i < _iterationLimit) {
      final gradient =
          _costFunction.getGradient(_features, coefficients, _labels);

      coefficients = coefficients - inverseHessian * gradient;
      i++;

      if (collectLearningData) {
        final cost = _costFunction.getCost(_features, coefficients, _labels);

        _costPerIteration.add(cost);
      }
    }

    return coefficients;
  }
}
