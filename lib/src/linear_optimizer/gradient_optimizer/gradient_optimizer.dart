import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/di/dependencies.dart';
import 'package:ml_algo/src/linear_optimizer/convergence_detector/convergence_detector.dart';
import 'package:ml_algo/src/linear_optimizer/convergence_detector/convergence_detector_factory.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_generator.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_generator_factory.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_generator.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_generator_factory.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer.dart';
import 'package:ml_algo/src/math/randomizer/randomizer.dart';
import 'package:ml_algo/src/math/randomizer/randomizer_factory.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:xrange/integers.dart';

class GradientOptimizer implements LinearOptimizer {
  GradientOptimizer(Matrix points, Matrix labels, {
    DType dtype = DType.float32,
    CostFunction costFunction,
    LearningRateType learningRateType = LearningRateType.decreasingAdaptive,
    InitialCoefficientsType initialCoefficientsType = InitialCoefficientsType.zeroes,
    double initialLearningRate = 1e-3,
    double minCoefficientsUpdate = 1e-12,
    int iterationLimit = 100,
    double lambda,
    int batchSize,
    int randomSeed,
  })  :
        _points = points,
        _labels = labels,
        _lambda = lambda ?? 0.0,
        _batchSize = batchSize,
        _costFunction = costFunction,
        _dtype = dtype,

        _initialCoefficientsGenerator = dependencies
            .getDependency<InitialCoefficientsGeneratorFactory>()
            .fromType(initialCoefficientsType, dtype),

        _learningRateGenerator = dependencies
            .getDependency<LearningRateGeneratorFactory>()
            .fromType(learningRateType),

        _convergenceDetector = dependencies
            .getDependency<ConvergenceDetectorFactory>()
            .create(minCoefficientsUpdate, iterationLimit),

        _randomizer = dependencies
            .getDependency<RandomizerFactory>()
            .create(randomSeed) {
    if (batchSize < 1 || batchSize > points.rowsNum) {
      throw RangeError.range(batchSize, 1, points.rowsNum, 'Invalid batch size '
          'value');
    }
    _learningRateGenerator.init(initialLearningRate ?? 1.0);
  }

  final Matrix _points;
  final Matrix _labels;
  final Randomizer _randomizer;
  final CostFunction _costFunction;
  final LearningRateGenerator _learningRateGenerator;
  final InitialCoefficientsGenerator _initialCoefficientsGenerator;
  final ConvergenceDetector _convergenceDetector;
  final DType _dtype;
  final double _lambda;
  final int _batchSize;
  final List<num> _costPerIteration = [];

  @override
  List<num> get costPerIteration => _costPerIteration;

  @override
  Matrix findExtrema({
    Matrix initialCoefficients,
    bool isMinimizingObjective = true,
    bool collectLearningData = false,
  }) {
    _costPerIteration.clear();

    var coefficients = initialCoefficients ??
        Matrix.fromColumns(List.generate(_labels.columnsNum,
            (i) => _initialCoefficientsGenerator.generate(_points.columnsNum)),
            dtype: _dtype);

    var iteration = 0;
    var coefficientsDiff = double.maxFinite;

    while (!_convergenceDetector.isConverged(coefficientsDiff, iteration)) {
      final learningRate = _learningRateGenerator.getNextValue();
      final newCoefficients = _generateCoefficients(
        coefficients,
        learningRate,
        isMinimization: isMinimizingObjective,
        collectLearningData: collectLearningData,
      );
      coefficientsDiff = (newCoefficients - coefficients).norm();
      iteration++;
      coefficients = newCoefficients;
    }

    _learningRateGenerator.stop();

    return coefficients;
  }

  /// [coefficients] columns of coefficients (each label columns has its own
  /// dedicated column of coefficients)
  Matrix _generateCoefficients(
    Matrix coefficients,
    double learningRate, {
      bool isMinimization = true,
      bool collectLearningData = false,
    }) {

    final range = _getBatchRange();
    final start = range.first;
    final end = range.last;
    final pointsBatch = _points
        .sample(rowIndices: integers(start, end, upperClosed: false));
    final labelsBatch = _labels
        .sample(rowIndices: integers(start, end, upperClosed: false));

    return _makeGradientStep(
      coefficients,
      pointsBatch,
      labelsBatch,
      learningRate,
      isMinimization: isMinimization,
      collectLearningData: collectLearningData,
    );
  }

  Iterable<int> _getBatchRange() => _randomizer
      .getIntegerInterval(0, _points.rowsNum, intervalLength: _batchSize);

  /// [coefficients] columns of coefficients (each label column from [labels]
  /// has its own dedicated column of coefficients)
  ///
  /// [labels] columns of labels
  Matrix _makeGradientStep(
      Matrix coefficients,
      Matrix points,
      Matrix labels,
      double eta,
      {
        bool isMinimization = true,
        bool collectLearningData = false,
      }) {

    if (collectLearningData) {
      final error = _costFunction.getCost(points, coefficients, labels);

      _costPerIteration.add(error);
    }

    final gradient = _costFunction.getGradient(points, coefficients, labels);
    final regularizedCoefficients = _regularize(eta, _lambda, coefficients);

    return isMinimization
        ? regularizedCoefficients - gradient * eta
        : regularizedCoefficients + gradient * eta;
  }

  Matrix _regularize(double learningRate, double lambda, Matrix coefficients) {
    if (lambda == 0) {
      return coefficients;
    }

    return coefficients * (1 - 2 * learningRate * lambda);
  }
}
