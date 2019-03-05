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

  Matrix _coefficients;
  Vector _normalizer;

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
  Matrix findExtrema(Matrix points, Matrix labels,
      {
        Matrix initialWeights,
        bool isMinimizingObjective = true,
        bool arePointsNormalized = false
      }
  ) {
    _normalizer = arePointsNormalized
        ? Vector.filled(points.columnsNum, 1.0, dtype: _dtype)
        : points.reduceRows((combine, vector) => (combine + vector * vector));

    _coefficients = initialWeights ??
        Matrix.rows(List<Vector>.generate(labels.columnsNum,
                (int i) => _initialCoefficientsGenerator
                    .generate(points.columnsNum)));

    int iteration = 0;
    var diff = double.infinity;
    while (!_convergenceDetector.isConverged(diff, iteration)) {
      final newCoefsSource = List<Vector>(points.columnsNum);
      for (int j = 0; j < points.columnsNum; j++) {
        final jCoeffs = _coefficients.getColumn(j);
        final newJCoeffs = _optimizeCoordinate(j, points, labels,
            _coefficients);
        // TODO improve diff calculation way
        // Now we just get maximum diff throughout the whole coefficients
        // vector and compare it with some limit (inside _convergenceDetector)
        diff = (jCoeffs - newJCoeffs).abs().max();
        newCoefsSource[j] = newJCoeffs;
      }
      // TODO: get rid of redundant matrix creation
      _coefficients = Matrix.columns(newCoefsSource);
      iteration++;
    }

    return _coefficients;
  }

  Vector _optimizeCoordinate(int j, Matrix x, Matrix y, Matrix w) {
    // coefficients variable here contains coefficients on column j per each
    // label
    final coefficients = _costFn.getSubDerivative(j, x, w, y);
    return Vector
        // TODO Convert the logic into SIMD-way (SIMD way mapping)
        .from(coefficients.map((coef) => _regularize(coef, _lambda, j)));
  }

  double _regularize(double coefficient, double lambda, int coefNum) {
    if (lambda == 0.0) {
      return coefficient;
    }
    final threshold = lambda / 2;
    if (coefficient > threshold) {
      return (coefficient - threshold) / _normalizer[coefNum];
    }
    if (coefficient < -threshold) {
      return (coefficient + threshold) / _normalizer[coefNum];
    }
    return 0.0;
  }
}
