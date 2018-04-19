import 'dart:typed_data';

import 'package:dart_ml/src/loss_function/loss_function.dart';
import 'package:dart_ml/src/math/math_analysis/gradient_calculator.dart';
import 'package:dart_ml/src/math/randomizer/randomizer.dart';
import 'package:dart_ml/src/optimizer/learning_rate_generator/learning_rate_generator.dart';
import 'package:dart_ml/src/optimizer/initial_weights_generator/initial_weights_generator.dart';
import 'package:dart_ml/src/optimizer/optimizer.dart';
import 'package:simd_vector/vector.dart';

class GradientOptimizer implements Optimizer {

  final Randomizer _randomizer;
  final LossFunction _lossFunction;
  final GradientCalculator _gradientCalculator;
  final LearningRateGenerator _learningRateGenerator;
  final InitialWeightsGenerator _initialWeightsGenerator;

  //hyper parameters declaration
  final double _minCoefficientsUpdate;
  final int _iterationLimit;
  final double _lambda;
  final double _argumentIncrement;
  //hyper parameters declaration end

  final int _batchSize;

  List<Float32x4Vector> _points;

  GradientOptimizer(
    this._randomizer,
    this._lossFunction,
    this._gradientCalculator,
    this._learningRateGenerator,
    this._initialWeightsGenerator,
    {
      double learningRate,
      double minCoefficientsUpdate,
      int iterationLimit,
      double lambda,
      double argumentIncrement,
      int batchSize
    }
  ) :
    _minCoefficientsUpdate = minCoefficientsUpdate ?? 1e-8,
    _iterationLimit = iterationLimit ?? 10000,
    _lambda = lambda ?? 1e-5,
    _argumentIncrement = argumentIncrement ?? 1e-5,
    _batchSize = batchSize
  {
    _learningRateGenerator.init(learningRate ?? 1e-5);
  }

  @override
  Float32x4Vector findExtrema(
    covariant List<Float32x4Vector> points,
    covariant Float32List labels,
    {
      covariant Float32x4Vector initialWeights,
      bool isMinimizingObjective = true
    }
  ) {
    _points = points;

    Float32x4Vector coefficients = initialWeights ?? _initialWeightsGenerator.generate(_points.first.length);
    double coefficientsUpdate = double.MAX_FINITE;
    int iterationCounter = 0;

    while (coefficientsUpdate > _minCoefficientsUpdate && iterationCounter++ < _iterationLimit) {
      final eta = _learningRateGenerator.getNextValue();
      final updatedCoefficients = _generateCoefficients(coefficients, labels, eta, isMinimization: isMinimizingObjective);
      coefficientsUpdate = updatedCoefficients.distanceTo(coefficients);
      coefficients = updatedCoefficients;
    }

    _learningRateGenerator.stop();

    return coefficients;
  }

  Float32x4Vector _generateCoefficients(
    Float32x4Vector currentCoefficients,
    Float32List labels,
    double eta,
    {bool isMinimization: true}
  ) {
    final range = _getBatchRange();
    final start = range.first;
    final end = range.last;
    final pointsBatch = _points.sublist(start, end);
    final labelsBatch = labels.sublist(start, end);

    return _makeGradientStep(currentCoefficients, pointsBatch, labelsBatch, eta, isMinimization: isMinimization);
  }

  Iterable<int> _getBatchRange() => _randomizer.getIntegerInterval(0, _points.length, intervalLength: _batchSize);

  Float32x4Vector _makeGradientStep(
    Float32x4Vector coefficients,
    List<Float32x4Vector> points,
    Float32List labels,
    double eta,
    {bool isMinimization: true}
  ) {
    Float32x4Vector gradient = _getGradient(coefficients, points[0], labels[0]);
    for (int i = 1; i < points.length; i++) {
      gradient += _getGradient(coefficients, points[i], labels[i]);
    }
    return isMinimization ? coefficients - gradient.scalarMul(eta) : coefficients + gradient.scalarMul(eta);
  }

  Float32x4Vector _getGradient(
    Float32x4Vector k,
    Float32x4Vector x,
    double y
  ) {
    final gradient = _gradientCalculator.getGradient(
      (Vector k, Iterable<Vector> vectorArgs, Iterable<double> scalarArgs) {
        final x = (vectorArgs as List<Float32x4Vector>)[0];
        final y = (scalarArgs as List<double>)[0];
        final lambda = (scalarArgs as List<double>)[1];
        return _lossFunction.loss(k.dot(x), y) + lambda * k.norm(Norm.EUCLIDEAN);
      }, k, [x], [y, _lambda], _argumentIncrement
    );

    return gradient;
  }
}
