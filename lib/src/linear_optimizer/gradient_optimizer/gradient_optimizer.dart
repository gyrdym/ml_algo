import 'package:ml_algo/src/common/constants/default_parameters/common.dart';
import 'package:ml_algo/src/common/constants/default_parameters/gradient_optimization.dart';
import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_generator.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer.dart';
import 'package:ml_algo/src/math/randomizer/randomizer.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:xrange/xrange.dart';

class GradientOptimizer implements LinearOptimizer {
  GradientOptimizer(
    Matrix points,
    Matrix labels, {
    DType dtype = dTypeDefaultValue,
    required InitialCoefficientsGenerator initialCoefficientsGenerator,
    required CostFunction costFunction,
    required Iterable<double> learningRates,
    required double minCoefficientsUpdate,
    required Randomizer randomizer,
    required int batchSize,
    double? lambda,
  })  : _points = points,
        _labels = labels,
        _lambda = lambda ?? lambdaDefaultValue,
        _batchSize = batchSize,
        _costFunction = costFunction,
        _dtype = dtype,
        _initialCoefficientsGenerator = initialCoefficientsGenerator,
        _learningRates = learningRates,
        _minCoefficientsUpdate = minCoefficientsUpdate,
        _randomizer = randomizer {
    if (batchSize < 1 || batchSize > points.rowsNum) {
      throw RangeError.range(
          batchSize,
          1,
          points.rowsNum,
          'Invalid batch size '
          'value');
    }
  }

  final Matrix _points;
  final Matrix _labels;
  final Randomizer _randomizer;
  final CostFunction _costFunction;
  final Iterable<double> _learningRates;
  final InitialCoefficientsGenerator _initialCoefficientsGenerator;
  final double _minCoefficientsUpdate;
  final DType _dtype;
  final double _lambda;
  final int _batchSize;
  final List<num> _costPerIteration = [];

  @override
  List<num> get costPerIteration => _costPerIteration;

  @override
  Matrix findExtrema({
    Matrix? initialCoefficients,
    bool isMinimizingObjective = true,
    bool collectLearningData = false,
  }) {
    _costPerIteration.clear();

    var coefficients = initialCoefficients ??
        Matrix.fromColumns(
          List.generate(
            _labels.columnsNum,
            (i) => _initialCoefficientsGenerator.generate(_points.columnsNum),
          ),
          dtype: _dtype,
        );

    var coefficientsDiff = double.maxFinite;

    for (final learningRate in _learningRates) {
      if (coefficientsDiff <= _minCoefficientsUpdate) {
        break;
      }

      final newCoefficients = _generateCoefficients(
        coefficients,
        learningRate,
        isMinimization: isMinimizingObjective,
        collectLearningData: collectLearningData,
      );

      coefficientsDiff = (newCoefficients - coefficients).norm();
      coefficients = newCoefficients;
    }

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
    final pointsBatch = _points.sample(rowIndices: integers(start, end));
    final labelsBatch = _labels.sample(rowIndices: integers(start, end));

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
    double learningRate, {
    bool isMinimization = true,
    bool collectLearningData = false,
  }) {
    if (collectLearningData) {
      final error = _costFunction.getCost(points, coefficients, labels);

      _costPerIteration.add(error);
    }

    final gradient = _costFunction.getGradient(
      points,
      coefficients,
      labels,
    );
    final regularizedCoefficients = _regularize(
      learningRate,
      _lambda,
      coefficients,
    );

    return isMinimization
        ? regularizedCoefficients - gradient * learningRate
        : regularizedCoefficients + gradient * learningRate;
  }

  Matrix _regularize(double learningRate, double lambda, Matrix coefficients) {
    if (lambda == 0) {
      return coefficients;
    }

    return coefficients * (1 - 2 * learningRate * lambda);
  }
}
