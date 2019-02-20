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
  MLMatrix findExtrema(MLMatrix points, MLMatrix labels,
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

    int iteration = 0;
    final diffs = MLVector.filled(numOfCoefficientVectors, double.infinity,
        isMutable: true, dtype: _dtype);
    while (!_convergenceDetector.isConverged(diffs.max(), iteration)) {
      for (int j = 0; j < points.columnsNum; j++) {
        final jCoeffs = _coefficients.getColumn(j);
        final newJCoeffs = _optimizeCoordinate(j, points, labels,
            _coefficients);
        // TODO improve diff calculation way
        diffs[j] = jCoeffs.distanceTo(newJCoeffs);
        _coefficients.setColumn(j, jCoeffs);
      }
      iteration++;
    }

    return _coefficients;
  }

  MLVector _optimizeCoordinate(int coordIdx, MLMatrix x, MLMatrix y, MLMatrix w) {
    final coefficients = _costFn.getSubDerivative(coordIdx, x, w, y);
    return MLVector
        // TODO Convert the logic into SIMD-way (SIMD way mapping)
        .from(coefficients.map((coef) => _regularize(coef, _lambda, coordIdx)));
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
