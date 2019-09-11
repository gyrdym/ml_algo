import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/solver/linear/convergence_detector/convergence_detector.dart';
import 'package:ml_algo/src/solver/linear/convergence_detector/convergence_detector_factory.dart';
import 'package:ml_algo/src/solver/linear/convergence_detector/convergence_detector_factory_impl.dart';
import 'package:ml_algo/src/solver/linear/initial_weights_generator/initial_weights_generator.dart';
import 'package:ml_algo/src/solver/linear/initial_weights_generator/initial_weights_generator_factory.dart';
import 'package:ml_algo/src/solver/linear/initial_weights_generator/initial_weights_generator_factory_impl.dart';
import 'package:ml_algo/src/solver/linear/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/solver/linear/linear_optimizer.dart';
import 'package:ml_algo/src/utils/default_parameter_values.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/linalg.dart';

class CoordinateOptimizer implements LinearOptimizer {
  CoordinateOptimizer(Matrix points, Matrix labels, {
    DType dtype = DefaultParameterValues.dtype,
    CostFunction costFunction,
    InitialWeightsGeneratorFactory initialWeightsGeneratorFactory =
        const InitialWeightsGeneratorFactoryImpl(),
    ConvergenceDetectorFactory convergenceDetectorFactory =
        const ConvergenceDetectorFactoryImpl(),
    double minCoefficientsDiff = DefaultParameterValues.minCoefficientsUpdate,
    int iterationsLimit = DefaultParameterValues.iterationsLimit,
    double lambda,
    InitialWeightsType initialWeightsType = InitialWeightsType.zeroes,
    bool isTrainDataNormalized = false,
  })  : _dtype = dtype,
        _points = points,
        _labels = labels,
        _lambda = lambda ?? 0.0,
        _initialCoefficientsGenerator =
            initialWeightsGeneratorFactory.fromType(initialWeightsType, dtype),
        _convergenceDetector = convergenceDetectorFactory.create(
            minCoefficientsDiff, iterationsLimit),
        _costFn = costFunction,
        _normalizer = isTrainDataNormalized
            ? Vector.filled(points.columnsNum, 1.0, dtype: dtype)
            : points.reduceRows(
                (combine, vector) => (combine + vector * vector));

  final Matrix _points;
  final Matrix _labels;
  final InitialWeightsGenerator _initialCoefficientsGenerator;
  final ConvergenceDetector _convergenceDetector;
  final CostFunction _costFn;
  final DType _dtype;
  final double _lambda;
  final Vector _normalizer;

  @override
  Matrix findExtrema({
    Matrix initialWeights,
    bool isMinimizingObjective = true,
  }) {
    Matrix coefficients = initialWeights ??
        Matrix.fromRows(List<Vector>.generate(_labels.columnsNum,
                (int i) => _initialCoefficientsGenerator
                    .generate(_points.columnsNum)), dtype: _dtype);

    int iteration = 0;
    var diff = double.infinity;
    while (!_convergenceDetector.isConverged(diff, iteration)) {
      final newCoefsSource = List<Vector>(_points.columnsNum);
      for (int j = 0; j < _points.columnsNum; j++) {
        final jCoeffs = coefficients.getColumn(j);
        final newJCoeffs = _optimizeCoordinate(j, _points, _labels,
            coefficients);
        // TODO improve diff calculation way
        // Now we just get maximum diff throughout the whole coefficients
        // vector and compare it with some limit (inside _convergenceDetector)
        diff = (jCoeffs - newJCoeffs).abs().max();
        newCoefsSource[j] = newJCoeffs;
      }
      // TODO: get rid of redundant matrix creation
      coefficients = Matrix.fromColumns(newCoefsSource, dtype: _dtype);
      iteration++;
    }

    return coefficients;
  }

  Vector _optimizeCoordinate(int j, Matrix x, Matrix y, Matrix w) {
    // coefficients variable here contains coefficients on column j per each
    // label
    final coefficients = _costFn.getSubDerivative(j, x, w, y);
    return Vector
        // TODO Convert the logic into SIMD-way (SIMD way mapping)
        .fromList(coefficients.map((coef) => _regularize(coef, _lambda, j))
        .toList(growable: false));
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
