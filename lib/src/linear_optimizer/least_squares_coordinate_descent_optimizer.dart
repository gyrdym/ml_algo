import 'package:ml_algo/src/linear_optimizer/convergence_detector/convergence_detector.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_generator.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class LeastSquaresCoordinateDescentOptimizer implements LinearOptimizer {
  LeastSquaresCoordinateDescentOptimizer(
    Matrix features,
    Matrix labels, {
    required DType dtype,
    required ConvergenceDetector convergenceDetector,
    required double lambda,
    required InitialCoefficientsGenerator initialCoefficientsGenerator,
    required bool isFittingDataNormalized,
  })  : _dtype = dtype,
        _features = features,
        _labels = labels,
        _lambda = lambda,
        _featureMatrices = features.columnIndices
            .map((j) => features.filterColumns((_, idx) => idx != j))
            .toList(growable: false),
        _featureColumns = features.columns.toList(growable: false),
        _initialCoefficientsGenerator = initialCoefficientsGenerator,
        _convergenceDetector = convergenceDetector,
        _normalizer = isFittingDataNormalized
            ? Vector.filled(features.columnsNum, 1.0, dtype: dtype)
            : features
                .reduceRows((combine, vector) => (combine + vector * vector));

  final Matrix _features;
  final Matrix _labels;
  final InitialCoefficientsGenerator _initialCoefficientsGenerator;
  final ConvergenceDetector _convergenceDetector;
  final DType _dtype;
  final double _lambda;
  final Vector _normalizer;
  final List<num> _errors = [];
  // Feature matrices prepared for every iteration
  final List<Matrix> _featureMatrices;
  final List<Vector> _featureColumns;

  @override
  List<num> get costPerIteration => _errors;

  @override
  Matrix findExtrema({
    Matrix? initialCoefficients,
    bool isMinimizingObjective = true,
    bool collectLearningData = false,
  }) {
    var coefficients = initialCoefficients?.toVector() ??
        _initialCoefficientsGenerator.generate(_features.columnsNum);
    var labelsAsVector = _labels.toVector();

    var iteration = 0;
    var diff = double.infinity;

    while (!_convergenceDetector.isConverged(diff, iteration)) {
      final newCoeffsSource = List.generate(_features.columnsNum, (j) {
        final jCoef = coefficients[j];
        final newJCoef =
            _optimizeCoordinate(j, _features, labelsAsVector, coefficients);

        diff = (jCoef - newJCoef).abs();

        return newJCoef;
      });

      coefficients = Vector.fromList(newCoeffsSource, dtype: _dtype);
      iteration++;
    }

    return Matrix.fromColumns([coefficients], dtype: _dtype);
  }

  double _optimizeCoordinate(int j, Matrix X, Vector y, Vector w) {
    final xj = _featureColumns[j];
    final XWithoutJ = _featureMatrices[j];
    final wWithoutJ = w.filterElements((_, idx) => idx != j);
    final coef = (xj * (y - XWithoutJ * wWithoutJ)).sum();

    return _regularize(coef, _lambda, j);
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
