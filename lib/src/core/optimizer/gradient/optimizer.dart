import 'dart:typed_data';

import 'package:dart_ml/src/core/loss_function/loss_function.dart';
import 'package:dart_ml/src/core/math/math_analysis/gradient_calculator.dart';
import 'package:dart_ml/src/core/math/randomizer/randomizer.dart';
import 'package:dart_ml/src/core/optimizer/gradient/initial_weights_generator/initial_weights_generator.dart';
import 'package:dart_ml/src/core/optimizer/gradient/learning_rate_generator/learning_rate_generator.dart';
import 'package:dart_ml/src/core/optimizer/optimizer.dart';
import 'package:dart_ml/src/core/score_function/score_function.dart';
import 'package:dart_ml/src/di/injector.dart';
import 'package:simd_vector/vector.dart';

class GradientOptimizerImpl implements Optimizer {

  final Randomizer _randomizer = coreInjector.get(Randomizer);
  final LossFunction _lossFunction = coreInjector.get(LossFunction);
  final ScoreFunction _scoreFunction = coreInjector.get(ScoreFunction);
  final GradientCalculator _gradientCalculator = coreInjector.get(GradientCalculator);
  final LearningRateGenerator _learningRateGenerator = coreInjector.get(LearningRateGenerator);
  final InitialWeightsGenerator _initialWeightsGenerator = coreInjector.get(InitialWeightsGenerator);

  //hyper parameters declaration
  final double _minCoefficientsUpdate;
  final int _iterationLimit;
  final double _lambda;
  final double _argumentIncrement;
  //hyper parameters declaration end

  final int _batchSize;

  List<Float32x4Vector> _points;

  GradientOptimizerImpl({
    double learningRate,
    double minCoefficientsUpdate,
    int iterationLimit,
    double lambda,
    double argumentIncrement,
    int batchSize
  }) :
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
      (Float32x4Vector k, List<Float32x4Vector> vectorArgs, List<double> scalarArgs) {
        final x = vectorArgs[0];
        final y = scalarArgs[0];
        final lambda = scalarArgs[1];
        return _lossFunction.loss(_scoreFunction.score(k, x), y) + lambda * k.norm(Norm.EUCLIDEAN);
      }, k, [x], [y, _lambda], _argumentIncrement
    );

    return gradient;
  }
}
