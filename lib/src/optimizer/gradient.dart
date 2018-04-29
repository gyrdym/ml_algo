import 'package:dart_ml/src/cost_function/cost_function.dart';
import 'package:dart_ml/src/math/randomizer/randomizer.dart';
import 'package:dart_ml/src/optimizer/initial_weights_generator/initial_weights_generator.dart';
import 'package:dart_ml/src/optimizer/learning_rate_generator/learning_rate_generator.dart';
import 'package:dart_ml/src/optimizer/optimizer.dart';
import 'package:simd_vector/vector.dart';

class GradientOptimizer implements Optimizer {

  final Randomizer _randomizer;
  final CostFunction _costFunction;
  final LearningRateGenerator _learningRateGenerator;
  final InitialWeightsGenerator _initialWeightsGenerator;

  //hyper parameters declaration
  final double _minCoefficientsUpdate;
  final int _iterationLimit;
  final double _lambda;
  //hyper parameters declaration end

  final int _batchSize;

  List<Float32x4Vector> _points;

  GradientOptimizer(
    this._randomizer,
    this._costFunction,
    this._learningRateGenerator,
    this._initialWeightsGenerator,
    {
      double learningRate,
      double minCoefficientsUpdate,
      int iterationLimit,
      double lambda,
      int batchSize
    }
  ) :
    _minCoefficientsUpdate = minCoefficientsUpdate ?? 1e-8,
    _iterationLimit = iterationLimit ?? 10000,
    _lambda = lambda ?? 1e-5,
    _batchSize = batchSize
  {
    _learningRateGenerator.init(learningRate ?? 1e-5);
  }

  @override
  Float32x4Vector findExtrema(
    covariant List<Float32x4Vector> points,
    covariant Float32x4Vector labels,
    {
      covariant Float32x4Vector initialWeights,
      bool isMinimizingObjective = true,
      bool arePointsNormalized = false
    }
  ) {
    _points = points;

    Float32x4Vector coefficients = initialWeights ?? _initialWeightsGenerator.generate(_points.first.length);
    double coefficientsUpdate = double.MAX_FINITE;
    int iterationCounter = 0;

    while (!_isConverged(coefficientsUpdate, iterationCounter)) {
      final eta = _learningRateGenerator.getNextValue();
      final updatedCoefficients = _generateCoefficients(coefficients, labels, eta, isMinimization: isMinimizingObjective);
      coefficientsUpdate = updatedCoefficients.distanceTo(coefficients);
      coefficients = updatedCoefficients;
    }

    _learningRateGenerator.stop();

    return coefficients;
  }

  bool _isConverged(double coefficientsUpdate, int iterationCounter) {
    return
      _minCoefficientsUpdate != null ? coefficientsUpdate < _minCoefficientsUpdate : false ||
      _iterationLimit != null ? iterationCounter++ >= _iterationLimit : false;
  }

  Float32x4Vector _generateCoefficients(
    Float32x4Vector currentCoefficients,
    Float32x4Vector labels,
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
    Float32x4Vector labels,
    double eta,
    {bool isMinimization: true}
  ) {
    Float32x4Vector gradient = new Float32x4Vector.from(new List.generate(coefficients.length,
      (int j) =>
        _costFunction.getPartialDerivative(j, points[0], coefficients, labels[0]) + _lambda * 2 * coefficients[j]
    ));

    for (int i = 1; i < points.length; i++) {
      gradient += new Float32x4Vector.from(new List.generate(coefficients.length,
        (int j) =>
          _costFunction.getPartialDerivative(j, points[i], coefficients, labels[i]) + _lambda * 2 * coefficients[j]
      ));
    }

    return isMinimization ? coefficients - gradient.scalarMul(eta) : coefficients + gradient.scalarMul(eta);
  }
}
