import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/math/randomizer/randomizer.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_generator.dart';
import 'package:ml_algo/src/optimizer/learning_rate_generator/generator.dart';
import 'package:ml_algo/src/optimizer/optimizer.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/range.dart';
import 'package:ml_linalg/vector.dart';

class GradientOptimizer<T> implements Optimizer<T> {
  final Randomizer _randomizer;
  final CostFunction<T> _costFunction;
  final LearningRateGenerator _learningRateGenerator;
  final InitialWeightsGenerator<T> _initialWeightsGenerator;

  //hyper parameters declaration
  final double _minCoefficientsUpdate;
  final int _iterationLimit;
  final double _lambda;
  final int _batchSize;
  //hyper parameters declaration end

  MLMatrix<T> _points;

  GradientOptimizer(this._randomizer, this._costFunction, this._learningRateGenerator, this._initialWeightsGenerator,
      {double initialLearningRate, double minCoefficientsUpdate, int iterationLimit, double lambda, int batchSize})
      : _minCoefficientsUpdate = minCoefficientsUpdate,
        _iterationLimit = iterationLimit ?? 1000,
        _lambda = lambda ?? 0.0,
        _batchSize = batchSize {
    _learningRateGenerator.init(initialLearningRate ?? 1.0);
  }

  @override
  MLVector<T> findExtrema(MLMatrix<T> points, MLVector<T> labels,
      {MLVector<T> initialWeights, bool isMinimizingObjective = true, bool arePointsNormalized = false}) {
    _points = points;

    final batchSize = _batchSize >= _points.rowsNum ? _points.rowsNum : _batchSize;
    var coefficients = initialWeights ?? _initialWeightsGenerator.generate(_points.columnsNum);
    var coefficientsUpdate = double.maxFinite;
    var iterationCounter = 0;

    while (!_isConverged(coefficientsUpdate, iterationCounter)) {
      final eta = _learningRateGenerator.getNextValue();
      final updatedCoefficients =
          _generateCoefficients(coefficients, labels, eta, batchSize, isMinimization: isMinimizingObjective);
      coefficientsUpdate = updatedCoefficients.distanceTo(coefficients);
      coefficients = updatedCoefficients;
      iterationCounter++;
    }

    _learningRateGenerator.stop();

    return coefficients;
  }

  bool _isConverged(double coefficientsUpdate, int iterationCounter) =>
      (_minCoefficientsUpdate != null ? coefficientsUpdate <= _minCoefficientsUpdate : false) ||
      (iterationCounter >= _iterationLimit);

  MLVector<T> _generateCoefficients(
      MLVector<T> currentCoefficients, MLVector<T> labels, double eta, int batchSize,
      {bool isMinimization = true}) {
    final range = _getBatchRange(batchSize);
    final start = range.first;
    final end = range.last;
    final pointsBatch = _points.submatrix(rows: Range(start, end));
    final labelsBatch = labels.subvector(start, end);

    return _makeGradientStep(currentCoefficients, pointsBatch, labelsBatch, eta, isMinimization: isMinimization);
  }

  Iterable<int> _getBatchRange(int batchSize) =>
      _randomizer.getIntegerInterval(0, _points.rowsNum, intervalLength: batchSize);

  MLVector<T> _makeGradientStep(
      MLVector<T> coefficients, MLMatrix<T> points, MLVector<T> labels, double eta,
      {bool isMinimization = true}) {
    final gradient = _costFunction.getGradient(points, coefficients, labels);
    final regularizedCoefficients = _regularize(eta, _lambda, coefficients);
    return isMinimization ? regularizedCoefficients - gradient * eta : regularizedCoefficients + gradient * eta;
  }

  MLVector<T> _regularize(double eta, double lambda, MLVector<T> coefficients) {
    if (lambda == 0) {
      return coefficients;
    } else {
      return coefficients * (1 - 2 * eta * lambda);
    }
  }
}
