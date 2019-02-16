import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/cost_function/cost_function_factory.dart';
import 'package:ml_algo/src/cost_function/cost_function_factory_impl.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/default_parameter_values.dart';
import 'package:ml_algo/src/optimizer/convergence_detector/convergence_detector.dart';
import 'package:ml_algo/src/optimizer/convergence_detector/convergence_detector_factory.dart';
import 'package:ml_algo/src/optimizer/convergence_detector/convergence_detector_factory_impl.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_generator.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_generator_factory.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_generator_factory_impl.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/optimizer/optimizer.dart';
import 'package:ml_linalg/linalg.dart';

class CoordinateOptimizer implements Optimizer {
  final InitialWeightsGenerator _initialCoefficientsGenerator;
  final ConvergenceDetector _convergenceDetector;
  final CostFunction _costFn;
  final Type _dtype;
  final double _lambda;

  MLMatrix _coefficients;
  MLVector _normalizer;

  CoordinateOptimizer({
    Type dtype = DefaultParameterValues.dtype,
    InitialWeightsGeneratorFactory initialWeightsGeneratorFactory =
        const InitialWeightsGeneratorFactoryImpl(),
    ConvergenceDetectorFactory convergenceDetectorFactory =
        const ConvergenceDetectorFactoryImpl(),
    CostFunctionFactory costFunctionFactory = const CostFunctionFactoryImpl(),
    double minCoefficientsDiff = DefaultParameterValues.minCoefficientsUpdate,
    int iterationsLimit = DefaultParameterValues.iterationsLimit,
    double lambda,
    InitialWeightsType initialWeightsType = InitialWeightsType.zeroes,
    CostFunctionType costFunctionType = CostFunctionType.squared,
  })  : _dtype = dtype,
        _lambda = lambda ?? 0.0,
        _initialCoefficientsGenerator =
            initialWeightsGeneratorFactory.fromType(initialWeightsType, dtype),
        _convergenceDetector = convergenceDetectorFactory.create(
            minCoefficientsDiff, iterationsLimit),
        _costFn = costFunctionFactory.fromType(costFunctionType);

  @override
  MLMatrix findExtrema(MLMatrix points, MLVector labels,
      {int numOfCoefficientVectors = 1,
      MLMatrix initialWeights,
      bool isMinimizingObjective = true,
      bool arePointsNormalized = false}) {
    _normalizer = arePointsNormalized
        ? MLVector.filled(points.columnsNum, 1.0, dtype: _dtype)
        : points.reduceRows((combine, vector) => (combine + vector * vector));

    if (initialWeights != null) {
      _coefficients = initialWeights;
    } else {
      final initialCoefSource = List<MLVector>.generate(numOfCoefficientVectors,
          (int i) => _initialCoefficientsGenerator.generate(points.columnsNum));
      _coefficients = MLMatrix.rows(initialCoefSource, dtype: _dtype);
    }

    final updates = MLVector.filled(numOfCoefficientVectors, double.infinity,
        isMutable: true, dtype: _dtype);
    int iteration = 0;

    while (!_convergenceDetector.isConverged(updates, iteration)) {
      final coefficientsSource = List<MLVector>(numOfCoefficientVectors);
      final diffs =
          MLVector.zero(points.columnsNum, isMutable: true, dtype: _dtype);

      for (int k = 0; k < numOfCoefficientVectors; k++) {
        final newCoeffs =
            MLVector.zero(points.columnsNum, isMutable: true, dtype: _dtype);
        final currCoeffs = _coefficients.getRow(k);

        for (int j = 0; j < points.columnsNum; j++) {
          final currCoef = currCoeffs[j];
          final newCoef = _optimizeCoordinate(j, points, labels, currCoeffs);
          diffs[j] = (currCoef - newCoef).abs();
          newCoeffs[j] = newCoef;
        }

        updates[k] = diffs.max();
        coefficientsSource[k] = newCoeffs;
      }
      // TODO: get rid of matrix instantiating here, use a list
      _coefficients = MLMatrix.rows(coefficientsSource, dtype: _dtype);
      iteration++;
    }

    return _coefficients;
  }

  double _optimizeCoordinate(int coefficientNum, MLMatrix points,
      MLVector labels, MLVector coefficients) {
    double coefficient = 0.0;
    for (int rowNum = 0; rowNum < points.rowsNum; rowNum++) {
      final point = points.getRow(rowNum);
      final output = labels[rowNum];
      coefficient += _costFn.getSparseSolutionPartial(
          coefficientNum, point, coefficients, output);
    }
    return _regularize(coefficient, _lambda, coefficientNum);
  }

  double _regularize(double coefficient, double lambda, int coefNum) {
    if (lambda == 0.0) {
      return coefficient;
    }
    final threshold = lambda / 2;
    double regularized = 0.0;
    if (coefficient > threshold) {
      regularized = (coefficient - threshold) / _normalizer[coefNum];
    } else if (coefficient < -threshold) {
      regularized = (coefficient + threshold) / _normalizer[coefNum];
    }
    return regularized;
  }
}
