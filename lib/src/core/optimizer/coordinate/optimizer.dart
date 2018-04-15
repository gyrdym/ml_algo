import 'dart:typed_data';

import 'package:dart_ml/src/core/optimizer/gradient/initial_weights_generator/initial_weights_generator.dart';
import 'package:dart_ml/src/core/optimizer/optimizer.dart';
import 'package:dart_ml/src/core/score_function/score_function.dart';
import 'package:dart_ml/src/di/injector.dart';
import 'package:simd_vector/vector.dart';

class CoordinateOptimizerImpl implements Optimizer {
  final ScoreFunction _scoreFunction = coreInjector.get(ScoreFunction);
  final InitialWeightsGenerator _initialCoefficientsGenerator = coreInjector.get(InitialWeightsGenerator);

  //hyper parameters declaration
  final double _coefficientsDiffThreshold;
  final int _iterationLimit;
  final double _lambda;
  //hyper parameters declaration end

  CoordinateOptimizerImpl({
    double minCoefficientsDiff,
    int iterationLimit,
    double lambda
  }) :
    _coefficientsDiffThreshold = minCoefficientsDiff ?? 1e-8,
    _iterationLimit = iterationLimit ?? 10000,
    _lambda = lambda ?? 0.0;

  @override
  Float32x4Vector findExtrema(
    List<Float32x4Vector> points,
    Float32List labels,
    {
      Float32x4Vector initialWeights,
      bool isMinimizingObjective = true
    }
  ) {
    Float32x4Vector coefficients = initialWeights ?? _initialCoefficientsGenerator.generate(points.first.length);

    double coefficientsDiff = double.INFINITY;
    int iteration = 0;

    while (coefficientsDiff > _coefficientsDiffThreshold && iteration < _iterationLimit) {
      final updatedCoefficients = new List<double>.filled(coefficients.length, 0.0, growable: false);

      for (int j = 0; j < coefficients.length; j++) {
        final coefficientsAsList = coefficients.asList();
        coefficientsAsList[j] = 0.0;
        final coefficientsWithoutJ = new Float32x4Vector.from(coefficientsAsList);

        for (int i = 0; i < points.length; i++) {
          final pointAsList = points[i].asList();
          final x = pointAsList[j];
          final y = labels[i];

          pointAsList[j] = 0.0;

          final pointWithoutJ = new Float32x4Vector.from(pointAsList);
          final yHat = _scoreFunction.score(coefficientsWithoutJ, pointWithoutJ);

          updatedCoefficients[j] += x * (y - yHat);
        }
      }

      final regularizedCoefficients = _regularize(updatedCoefficients, _lambda);
      final newCoefficients = new Float32x4Vector.from(regularizedCoefficients);

      coefficientsDiff = newCoefficients.distanceTo(coefficients);
      coefficients = newCoefficients;

      iteration++;
    }

    return coefficients;
  }

  List<double> _regularize(List<double> coefficients, double lambda) {
    if (lambda == 0.0) {
      return coefficients;
    }

    final regularized = new List<double>.filled(coefficients.length, 0.0, growable: false);

    for (int i = 0; i < coefficients.length; i++) {
      double coefficient = coefficients[i];
      double delta = lambda / 2;

      if (coefficient > delta) {
        coefficient -= delta;
      } else if (coefficient < -delta) {
        coefficient += delta;
      } else {
        coefficient = 0.0;
      }

      regularized[i] = coefficient;
     }

     return regularized;
  }
}