import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:xrange/xrange.dart';

class LeastSquaresNewtonOptimizer implements LinearOptimizer {
  LeastSquaresNewtonOptimizer(
      {required Matrix features,
      required Matrix labels,
      required CostFunction costFunction,
      required int iterationLimit,
      required num minCoefficientsUpdate,
      num lambda = 0})
      : _features = features,
        _labels = labels,
        _costFunction = costFunction,
        _iterations = integers(0, iterationLimit),
        _minCoefficientsUpdate = minCoefficientsUpdate,
        _lambda = lambda;

  final Matrix _features;
  final Matrix _labels;
  final CostFunction _costFunction;
  final Iterable<int> _iterations;
  final List<num> _costPerIteration = [];
  final num _lambda;
  final num _minCoefficientsUpdate;

  @override
  List<num> get costPerIteration => _costPerIteration;

  @override
  Matrix findExtrema(
      {Matrix? initialCoefficients,
      bool isMinimizingObjective = true,
      bool collectLearningData = false}) {
    var dtype = _features.dtype;
    var coefficients = initialCoefficients ??
        Matrix.column(List.filled(_features.first.length, 0), dtype: dtype);
    var prevCoefficients = coefficients;
    var coefficientsUpdate = double.maxFinite;

    final regularizingTerm =
        Matrix.scalar(_lambda.toDouble(), _features.columnsNum, dtype: dtype);
    // Since we perfectly know that Hessian matrix calculation of least squares
    // function doesn't depend on coefficient vector, Hessian matrix will be
    // constant throughout the entire optimization procedure, let's calculate it
    // only once in the beginning of the procedure:
    final hessian = _costFunction.getHessian(_features, coefficients, _labels);
    final regularizedInverseHessian = _lambda == 0
        ? hessian.inverse()
        : (hessian + regularizingTerm).inverse();

    for (final _ in _iterations) {
      if (coefficientsUpdate.isNaN ||
          coefficientsUpdate <= _minCoefficientsUpdate) {
        break;
      }

      final gradient =
          _costFunction.getGradient(_features, coefficients, _labels);

      coefficients = coefficients - regularizedInverseHessian * gradient;
      coefficientsUpdate = (coefficients - prevCoefficients).norm();
      prevCoefficients = coefficients;

      if (collectLearningData) {
        final cost = _costFunction.getCost(_features, coefficients, _labels);

        _costPerIteration.add(cost);
      }
    }

    return coefficients;
  }
}
