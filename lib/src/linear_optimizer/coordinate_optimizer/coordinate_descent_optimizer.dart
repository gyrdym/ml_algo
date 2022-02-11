import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/linear_optimizer/convergence_detector/convergence_detector.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_generator.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer.dart';
import 'package:ml_linalg/linalg.dart';

class CoordinateDescentOptimizer implements LinearOptimizer {
  CoordinateDescentOptimizer(
    Matrix fittingPoints,
    Matrix fittingLabels, {
    required DType dtype,
    required CostFunction costFunction,
    required ConvergenceDetector convergenceDetector,
    required double lambda,
    required InitialCoefficientsGenerator initialCoefficientsGenerator,
    required bool isFittingDataNormalized,
  })  : _dtype = dtype,
        _points = fittingPoints,
        _labels = fittingLabels,
        _lambda = lambda,
        _initialCoefficientsGenerator = initialCoefficientsGenerator,
        _convergenceDetector = convergenceDetector,
        _costFn = costFunction,
        _normalizer = isFittingDataNormalized
            ? Vector.filled(fittingPoints.columnsNum, 1.0, dtype: dtype)
            : fittingPoints
                .reduceRows((combine, vector) => (combine + vector * vector));

  final Matrix _points;
  final Matrix _labels;
  final InitialCoefficientsGenerator _initialCoefficientsGenerator;
  final ConvergenceDetector _convergenceDetector;
  final CostFunction _costFn;
  final DType _dtype;
  final double _lambda;
  final Vector _normalizer;
  final List<num> _errors = [];

  @override
  List<num> get costPerIteration => _errors;

  @override
  Matrix findExtrema({
    Matrix? initialCoefficients,
    bool isMinimizingObjective = true,
    bool collectLearningData = false,
  }) {
    var coefficients = initialCoefficients ??
        Matrix.fromRows(
            List<Vector>.generate(
                _labels.columnsNum,
                (int i) =>
                    _initialCoefficientsGenerator.generate(_points.columnsNum)),
            dtype: _dtype);

    var iteration = 0;
    var diff = double.infinity;

    while (!_convergenceDetector.isConverged(diff, iteration)) {
      final newCoefsSource = List<Vector>.generate(_points.columnsNum, (j) {
        final jCoeffs = coefficients.getColumn(j);
        final newJCoeffs =
            _optimizeCoordinate(j, _points, _labels, coefficients);

        // TODO improve diff calculation way
        // Now we just get maximum diff throughout the whole coefficients
        // vector and compare it with some limit (inside _convergenceDetector)
        diff = (jCoeffs - newJCoeffs).abs().max();

        return newJCoeffs;
      });

      // TODO: get rid of redundant matrix creation
      coefficients = Matrix.fromColumns(newCoefsSource, dtype: _dtype);
      iteration++;
    }

    return coefficients.transpose();
  }

  Vector _optimizeCoordinate(int j, Matrix x, Matrix y, Matrix w) {
    // coefficients variable here contains coefficients on column j per each
    // label
    final coefficients = _costFn.getSubGradient(j, x, w, y);

    return Vector
        // TODO Convert the logic into SIMD-way (SIMD way mapping)
        .fromList(
            coefficients
                .map((coef) => _regularize(coef, _lambda, j))
                .toList(growable: false),
            dtype: _dtype);
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
