import 'dart:typed_data';

import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/math/randomizer/randomizer.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_generator.dart';
import 'package:ml_algo/src/optimizer/learning_rate_generator/generator.dart';
import 'package:ml_algo/src/optimizer/optimizer.dart';
import 'package:ml_linalg/linalg.dart';

class GradientOptimizer implements Optimizer<Float32x4, MLVector<Float32x4>> {

  final Randomizer _randomizer;
  final CostFunction<Float32x4> _costFunction;
  final LearningRateGenerator _learningRateGenerator;
  final InitialWeightsGenerator<Float32x4> _initialWeightsGenerator;

  //hyper parameters declaration
  final double _minCoefficientsUpdate;
  final int _iterationLimit;
  final double _lambda;
  final int _batchSize;
  //hyper parameters declaration end

  MLMatrix<Float32x4, MLVector<Float32x4>> _points;

  GradientOptimizer(
    this._randomizer,
    this._costFunction,
    this._learningRateGenerator,
    this._initialWeightsGenerator,
    {
      double initialLearningRate,
      double minCoefficientsUpdate,
      int iterationLimit,
      double lambda,
      int batchSize
    }
  ) :
    _minCoefficientsUpdate = minCoefficientsUpdate,
    _iterationLimit = iterationLimit ?? 1000,
    _lambda = lambda ?? 0.0,
    _batchSize = batchSize
  {
    _learningRateGenerator.init(initialLearningRate ?? 1.0);
  }

  @override
  MLVector<Float32x4> findExtrema(
    MLMatrix<Float32x4, MLVector<Float32x4>> points,
    MLVector<Float32x4> labels,
    {
      MLVector<Float32x4> initialWeights,
      bool isMinimizingObjective = true,
      bool arePointsNormalized = false
    }
  ) {
    _points = points;

    final batchSize = _batchSize >= _points.rowsNum ? _points.rowsNum : _batchSize;
    var coefficients = initialWeights ?? _initialWeightsGenerator.generate(_points.columnsNum);
    var coefficientsUpdate = double.maxFinite;
    var iterationCounter = 0;

    while (!_isConverged(coefficientsUpdate, iterationCounter)) {
      final eta = _learningRateGenerator.getNextValue();
      final updatedCoefficients = _generateCoefficients(coefficients, labels, eta, batchSize,
        isMinimization: isMinimizingObjective);
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

  MLVector<Float32x4> _generateCoefficients(
    MLVector<Float32x4> currentCoefficients,
    MLVector<Float32x4> labels,
    double eta,
    int batchSize,
    {bool isMinimization = true}
  ) {
    final range = _getBatchRange(batchSize);
    final start = range.first;
    final end = range.last;
    final pointsBatch = _points.submatrix(rows: Range(start, end));
    final labelsBatch = labels.subvector(start, end);

    return _makeGradientStep(currentCoefficients, pointsBatch, labelsBatch, eta, isMinimization: isMinimization);
  }

  Iterable<int> _getBatchRange(int batchSize) =>
      _randomizer.getIntegerInterval(0, _points.rowsNum, intervalLength: batchSize);

  MLVector<Float32x4> _makeGradientStep(
    MLVector<Float32x4> coefficients,
    MLMatrix<Float32x4, MLVector<Float32x4>> points,
    MLVector<Float32x4> labels,
    double eta,
    {bool isMinimization = true}
  ) {
    var gradient = Float32x4VectorFactory.zero(coefficients.length);
    for (var i = 0; i < points.rowsNum; i++) {
      final derivatives = List.generate(coefficients.length,
        (int j) => _costFunction.getPartialDerivative(j, points.getRowVector(i), coefficients, labels[i]));
      gradient += Float32x4VectorFactory.from(derivatives);
    }

    final regularizedCoefficients = _regularize(eta, _lambda, coefficients);
    return isMinimization ?
      regularizedCoefficients - gradient * eta :
      regularizedCoefficients + gradient * eta;
  }

  MLVector<Float32x4> _regularize(double eta, double lambda, MLVector<Float32x4> coefficients) {
    if (lambda == 0) {
      return coefficients;
    }

    return coefficients * (1 - 2 * eta * lambda);
  }
}
