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
  final int _batchSize;
  //hyper parameters declaration end

  @override
  CostFunction get costFunction => _costFunction;

  List<Float64x2Vector> _points;

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
    _minCoefficientsUpdate = minCoefficientsUpdate ?? 1e-8,
    _iterationLimit = iterationLimit ?? 10000,
    _lambda = lambda ?? 1e-5,
    _batchSize = batchSize
  {
    _learningRateGenerator.init(initialLearningRate ?? 1e-5);
  }

  @override
  Float64x2Vector findExtrema(
    covariant List<Float64x2Vector> points,
    covariant Float64x2Vector labels,
    {
      covariant Float64x2Vector initialWeights,
      bool isMinimizingObjective = true,
      bool arePointsNormalized = false
    }
  ) {

    final batchSize = _batchSize >= points.length ? points.length : _batchSize;

    _points = points;

    Float64x2Vector coefficients = initialWeights ?? _initialWeightsGenerator.generate(_points.first.length);
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
    (_iterationLimit != null ? iterationCounter >= _iterationLimit : false);


  Float64x2Vector _generateCoefficients(
    Float64x2Vector currentCoefficients,
    Float64x2Vector labels,
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

  Float64x2Vector _makeGradientStep(
    Float64x2Vector coefficients,
    List<Float64x2Vector> points,
    Float64x2Vector labels,
    double eta,
    {bool isMinimization: true}
  ) {
    Float64x2Vector gradient = new Float64x2Vector.from(new List.generate(coefficients.length,
      (int j) => _costFunction.getPartialDerivative(j, points[0], coefficients, labels[0])
    ));

    for (int i = 1; i < points.length; i++) {
      gradient += new Float64x2Vector.from(new List.generate(coefficients.length,
        (int j) => _costFunction.getPartialDerivative(j, points[i], coefficients, labels[i])
      ));
    }

    final regularizedCoefficients = _regularize(eta, _lambda, coefficients);
    return isMinimization ?
      regularizedCoefficients - gradient.scalarMul(eta) :
      regularizedCoefficients + gradient.scalarMul(eta);
  }

  Float64x2Vector _regularize(double eta, double lambda, Float64x2Vector coefficients) {
    if (lambda == 0) {
      return coefficients;
    }

    return coefficients.scalarMul(1 - 2 * eta * lambda);
  }
}
