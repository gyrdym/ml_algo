import 'package:ml_algo/learning_rate_type.dart';
import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/cost_function/cost_function_factory.dart';
import 'package:ml_algo/src/cost_function/cost_function_factory_impl.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/math/randomizer/randomizer.dart';
import 'package:ml_algo/src/math/randomizer/randomizer_factory.dart';
import 'package:ml_algo/src/math/randomizer/randomizer_factory_impl.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_generator.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_generator_factory.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_generator_factory_impl.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/optimizer/learning_rate_generator/learning_rate_generator.dart';
import 'package:ml_algo/src/optimizer/learning_rate_generator/learning_rate_generator_factory.dart';
import 'package:ml_algo/src/optimizer/learning_rate_generator/learning_rate_generator_factory_impl.dart';
import 'package:ml_algo/src/optimizer/optimizer.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/range.dart';
import 'package:ml_linalg/vector.dart';

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

  MLMatrix _points;

  GradientOptimizer({
    RandomizerFactory randomizerFactory = const RandomizerFactoryImpl(),
    CostFunctionFactory costFunctionFactory = const CostFunctionFactoryImpl(),
    LearningRateGeneratorFactory learningRateGeneratorFactory = const LearningRateGeneratorFactoryImpl(),
    InitialWeightsGeneratorFactory initialWeightsGeneratorFactory = const InitialWeightsGeneratorFactoryImpl(),
    CostFunctionType costFnType,
    LearningRateType learningRateType,
    InitialWeightsType initialWeightsType,
    Function scoreToProbLink,
    double initialLearningRate,
    double minCoefficientsUpdate,
    int iterationLimit,
    double lambda,
    int batchSize,
    int randomSeed,
  }) :
    _minCoefficientsUpdate = minCoefficientsUpdate,
    _iterationLimit = iterationLimit ?? 1000,
    _lambda = lambda ?? 0.0,
    _batchSize = batchSize,
    _initialWeightsGenerator = initialWeightsGeneratorFactory.fromType(initialWeightsType),
    _learningRateGenerator = learningRateGeneratorFactory.fromType(learningRateType),
    _costFunction = costFunctionFactory.fromType(costFnType, scoreToProbLink: scoreToProbLink),
    _randomizer = randomizerFactory.create(randomSeed) {
    _learningRateGenerator.init(initialLearningRate ?? 1.0);
  }

  @override
  MLVector findExtrema(MLMatrix points, MLVector labels,
      {MLVector initialWeights, bool isMinimizingObjective = true, bool arePointsNormalized = false}) {
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

  MLVector _generateCoefficients(
      MLVector currentCoefficients, MLVector labels, double eta, int batchSize,
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

  MLVector _makeGradientStep(
      MLVector coefficients, MLMatrix points, MLVector labels, double eta,
      {bool isMinimization = true}) {
    final gradient = _costFunction.getGradient(points, coefficients, labels);
    final regularizedCoefficients = _regularize(eta, _lambda, coefficients);
    return isMinimization ? regularizedCoefficients - gradient * eta : regularizedCoefficients + gradient * eta;
  }

  MLVector _regularize(double eta, double lambda, MLVector coefficients) {
    if (lambda == 0) {
      return coefficients;
    } else {
      return coefficients * (1 - 2 * eta * lambda);
    }
  }
}
