import 'package:dart_ml/src/cost_function/cost_function.dart';
import 'package:dart_ml/src/math/randomizer/randomizer.dart';
import 'package:dart_ml/src/optimizer/initial_weights_generator/initial_weights_generator.dart';
import 'package:dart_ml/src/optimizer/learning_rate_generator/generator.dart';
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
  final int _batchSize;
  //hyper parameters declaration end

  List<Float32x4Vector> _points;

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

    final batchSize = _batchSize >= _points.length ? _points.length : _batchSize;

    Float32x4Vector coefficients = initialWeights ?? _initialWeightsGenerator.generate(_points.first.length);
    double coefficientsUpdate = double.MAX_FINITE;
    int iterationCounter = 0;

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


  Float32x4Vector _generateCoefficients(
    Float32x4Vector currentCoefficients,
    Float32x4Vector labels,
    double eta,
    int batchSize,
    {bool isMinimization: true}
  ) {
    final range = _getBatchRange(batchSize);
    final start = range.first;
    final end = range.last;
    final pointsBatch = _points.sublist(start, end);
    final labelsBatch = labels.sublist(start, end);

    return _makeGradientStep(currentCoefficients, pointsBatch, labelsBatch, eta, isMinimization: isMinimization);
  }

  Iterable<int> _getBatchRange(int batchSize) => _randomizer.getIntegerInterval(0, _points.length, intervalLength: batchSize);

  Float32x4Vector _makeGradientStep(
    Float32x4Vector coefficients,
    List<Float32x4Vector> points,
    Float32x4Vector labels,
    double eta,
    {bool isMinimization: true}
  ) {
    Float32x4Vector gradient = new Float32x4Vector.from(new List.generate(coefficients.length,
      (int j) => _costFunction.getPartialDerivative(j, points[0], coefficients, labels[0])
    ));

    for (int i = 1; i < points.length; i++) {
      gradient += new Float32x4Vector.from(new List.generate(coefficients.length,
        (int j) => _costFunction.getPartialDerivative(j, points[i], coefficients, labels[i])
      ));
    }

    final regularizedCoefficients = _regularize(eta, _lambda, coefficients);
    return isMinimization ?
      regularizedCoefficients - gradient.scalarMul(eta) :
      regularizedCoefficients + gradient.scalarMul(eta);
  }

  Float32x4Vector _regularize(double eta, double lambda, Float32x4Vector coefficients) {
    if (lambda == 0) {
      return coefficients;
    }

    return coefficients.scalarMul(1 - 2 * eta * lambda);
  }
}
