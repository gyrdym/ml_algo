import 'dart:typed_data';

import 'package:dart_ml/src/cost_function/cost_function.dart';
import 'package:dart_ml/src/math/randomizer/randomizer.dart';
import 'package:dart_ml/src/optimizer/initial_weights_generator/initial_weights_generator.dart';
import 'package:dart_ml/src/optimizer/learning_rate_generator/generator.dart';
import 'package:dart_ml/src/optimizer/optimizer.dart';
import 'package:linalg/vector.dart';

class GradientOptimizer implements Optimizer<Float32x4> {

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

  List<Vector<Float32x4>> _points;

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
  Vector<Float32x4> findExtrema(
    List<Vector<Float32x4>> points,
    Vector<Float32x4> labels,
    {
      Vector<Float32x4> initialWeights,
      bool isMinimizingObjective = true,
      bool arePointsNormalized = false
    }
  ) {
    _points = points;

    final batchSize = _batchSize >= _points.length ? _points.length : _batchSize;
    var coefficients = initialWeights ?? _initialWeightsGenerator.generate(_points.first.length);
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


  Vector<Float32x4> _generateCoefficients(
    Vector<Float32x4> currentCoefficients,
    Vector<Float32x4> labels,
    double eta,
    int batchSize,
    {bool isMinimization = true}
  ) {
    final range = _getBatchRange(batchSize);
    final start = range.first;
    final end = range.last;
    final pointsBatch = _points.sublist(start, end);
    final labelsBatch = labels.subVector(start, end);

    return _makeGradientStep(currentCoefficients, pointsBatch, labelsBatch, eta, isMinimization: isMinimization);
  }

  Iterable<int> _getBatchRange(int batchSize) =>
      _randomizer.getIntegerInterval(0, _points.length, intervalLength: batchSize);

  Vector<Float32x4> _makeGradientStep(
    Vector<Float32x4> coefficients,
    List<Vector<Float32x4>> points,
    Vector<Float32x4> labels,
    double eta,
    {bool isMinimization = true}
  ) {
    var gradient = Float32x4VectorFactory.zero(coefficients.length);
    for (var i = 0; i < points.length; i++) {
      final derivatives = List.generate(coefficients.length,
        (int j) => _costFunction.getPartialDerivative(j, points[i], coefficients, labels[i]));
      gradient += Float32x4VectorFactory.from(derivatives);
    }

    final regularizedCoefficients = _regularize(eta, _lambda, coefficients);
    return isMinimization ?
      regularizedCoefficients - gradient.scalarMul(eta) :
      regularizedCoefficients + gradient.scalarMul(eta);
  }

  Vector<Float32x4> _regularize(double eta, double lambda, Vector<Float32x4> coefficients) {
    if (lambda == 0) {
      return coefficients;
    }

    return coefficients.scalarMul(1 - 2 * eta * lambda);
  }
}
