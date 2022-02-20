import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_algo/src/linear_optimizer/convergence_detector/convergence_detector.dart';
import 'package:ml_algo/src/linear_optimizer/convergence_detector/convergence_detector_impl.dart';
import 'package:ml_algo/src/linear_optimizer/coordinate_optimizer/least_squares_coordinate_descent_optimizer.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_generator.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/zero_coefficients_generator.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:test/test.dart';

import '../../helpers.dart';

/// L1 regularization, as known as Lasso, is aimed to penalize unimportant features, setting their weights to the zero,
/// therefore, we can treat the objective of the Lasso Optimizer like feature selection. Since lasso solver regularizes
/// coefficients by adding to their magnitude their L1-norm, we cannot use gradient methods any longer. Instead, we can
/// use coordinate descent optimization.

void main() {
  group('Coordinate descent solver (unregularized case)', () {
    const iterationsCount = 2;
    const lambda = 0.0;
    const dtype = DType.float32;
    const minCoefficientsUpdate = 1e-5;

    final point1 = [10.0, 20.0, 30.0];
    final point2 = [40.0, 50.0, 60.0];
    final point3 = [70.0, 80.0, 90.0];
    final point4 = [20.0, 30.0, 10.0];

    late InitialCoefficientsGenerator initialCoefficientsGenerator;
    late ConvergenceDetector convergenceDetector;
    late LeastSquaresCoordinateDescentOptimizer optimizer;
    late Matrix data;
    late Matrix labels;

    setUp(() {
      data = Matrix.fromList([point1, point2, point3, point4], dtype: dtype);
      labels = Matrix.fromList([
        [20.0],
        [30.0],
        [20.0],
        [40.0]
      ], dtype: dtype);

      initialCoefficientsGenerator = ZeroCoefficientsGenerator(dtype);
      convergenceDetector = ConvergenceDetectorImpl(
        minCoefficientsUpdate,
        iterationsCount,
      );
      optimizer = LeastSquaresCoordinateDescentOptimizer(
        data,
        labels,
        initialCoefficientsGenerator: initialCoefficientsGenerator,
        convergenceDetector: convergenceDetector,
        lambda: lambda,
        dtype: dtype,
        isFittingDataNormalized: false,
      );
    });

    tearDownAll(injector.clearAll);

    /// (The test case explanation)[https://github.com/gyrdym/ml_algo/wiki/Coordinate-descent-optimizer-(unregularized-case)-should-find-optimal-weights-for-the-given-data]
    test('should find optimal coefficients for the given data', () {
      final coefficients = optimizer.findExtrema();
      final expected = [-81796400.0, -81295300.0, -85285400.0];

      expect(coefficients.rowsNum, 3);
      expect(coefficients.columnsNum, 1);
      expect(coefficients.getColumn(0), iterableAlmostEqualTo(expected, 5.0));
    });
  });

  group('Coordinate descent solver (regularized case)', () {
    const iterationsCount = 2;
    const dtype = DType.float32;
    const lambda = 20.0; //define the regularization coefficient
    const minCoefficientsUpdate = 1e-5;

    final point1 = [10.0, 20.0, 30.0];
    final point2 = [20.0, 30.0, 40.0];
    final point3 = [70.0, 80.0, 90.0];

    late InitialCoefficientsGenerator initialCoefficientsGenerator;
    late ConvergenceDetector convergenceDetector;
    late LeastSquaresCoordinateDescentOptimizer optimizer;
    late Matrix data;
    late Matrix labels;

    setUp(() {
      data = Matrix.fromList([point1, point2, point3], dtype: dtype);
      labels = Matrix.fromList([
        [2.0],
        [3.0],
        [2.0],
      ], dtype: dtype);

      initialCoefficientsGenerator = ZeroCoefficientsGenerator(dtype);
      convergenceDetector = ConvergenceDetectorImpl(
        minCoefficientsUpdate,
        iterationsCount,
      );
      optimizer = LeastSquaresCoordinateDescentOptimizer(
        data,
        labels,
        isFittingDataNormalized: true,
        initialCoefficientsGenerator: initialCoefficientsGenerator,
        convergenceDetector: convergenceDetector,
        lambda: lambda,
        dtype: dtype,
      );
    });

    tearDown(injector.clearAll);

    /// (The test case explanation)[https://github.com/gyrdym/ml_algo/wiki/Coordinate-descent-optimizer-(regularized-case)-should-find-optimal-weights-for-the-given-data]
    test('should find optimal coefficients for the given data', () {
      // actually, points in this example are not normalized
      final coefficients = optimizer.findExtrema().getColumn(0);

      expect(coefficients, equals([-4381770, -4493700, -4073630]));
    });
  });
}
