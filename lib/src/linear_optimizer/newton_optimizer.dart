import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:xrange/xrange.dart';

class NewtonOptimizer implements LinearOptimizer {
  NewtonOptimizer(
      {required Matrix features,
      required Matrix labels,
      required CostFunction costFunction,
      required int iterationLimit,
      required num minCoefficientsUpdate,
      num lambda = 0,
      DType dtype = DType.float32})
      : _features = features,
        _labels = labels,
        _costFunction = costFunction,
        _iterations = integers(0, iterationLimit),
        _minCoefficientsUpdate = minCoefficientsUpdate,
        _lambda = lambda,
        _dtype = dtype;

  final Matrix _features;
  final Matrix _labels;
  final CostFunction _costFunction;
  final Iterable<int> _iterations;
  final List<num> _costPerIteration = [];
  final num _lambda;
  final num _minCoefficientsUpdate;
  final DType _dtype;

  @override
  List<num> get costPerIteration => _costPerIteration;

  @override
  Matrix findExtrema(
      {Matrix? initialCoefficients,
      bool isMinimizingObjective = true,
      bool collectLearningData = false}) {
    var coefficients = initialCoefficients ??
        Matrix.column(List.filled(_features.first.length, 0), dtype: _dtype);
    var prevCoefficients = coefficients;
    var coefficientsUpdate = double.maxFinite;

    final regularizingTerm =
        Matrix.scalar(_lambda.toDouble(), _features.columnsNum, dtype: _dtype);

    for (final _ in _iterations) {
      if (coefficientsUpdate.isNaN ||
          coefficientsUpdate <= _minCoefficientsUpdate) {
        break;
      }

      final gradient =
          _costFunction.getGradient(_features, coefficients, _labels);
      final hessian =
          _costFunction.getHessian(_features, coefficients, _labels);
      final regularizedInverseHessian = _lambda == 0
          ? hessian.inverse()
          : (hessian + regularizingTerm).inverse();

      coefficients = isMinimizingObjective
          ? coefficients - regularizedInverseHessian * gradient
          : coefficients + regularizedInverseHessian * gradient;
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
