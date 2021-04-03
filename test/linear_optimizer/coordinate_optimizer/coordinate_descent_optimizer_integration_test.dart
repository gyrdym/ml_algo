import 'package:ml_algo/src/cost_function/least_square_cost_function.dart';
import 'package:ml_algo/src/di/common/init_common_module.dart';
import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_algo/src/linear_optimizer/coordinate_optimizer/coordinate_descent_optimizer.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:test/test.dart';

import '../../helpers.dart';

/// L1 regularization, as known as Lasso, is aimed to penalize unimportant features, setting their weights to the zero,
/// therefore, we can treat the objective of the Lasso Optimizer like feature selection. Since lasso solver regularizes
/// coefficients by adding to their magnitude their L1-norm, we cannot use gradient methods any longer. Instead, we can
/// use coordinate descent optimization.

void main() {
  group('Coordinate descent solver (unregularized case)', () {
    const iterationsNumber = 2;
    const lambda = 0.0;

    final point1 = [10.0, 20.0, 30.0];
    final point2 = [40.0, 50.0, 60.0];
    final point3 = [70.0, 80.0, 90.0];
    final point4 = [20.0, 30.0, 10.0];

    late CoordinateDescentOptimizer optimizer;
    late Matrix data;
    late Matrix labels;

    setUp(() {
      initCommonModule();

      data = Matrix.fromList([point1, point2, point3, point4]);
      labels = Matrix.fromList([
        [20.0],
        [30.0],
        [20.0],
        [40.0]
      ]);
      optimizer = CoordinateDescentOptimizer(
        data,
        labels,
        initialWeightsType: InitialCoefficientsType.zeroes,
        costFunction: const LeastSquareCostFunction(),
        minCoefficientsUpdate: 1e-5,
        iterationsLimit: iterationsNumber,
        lambda: lambda,
        dtype: DType.float32,
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
    const iterationsNumber = 2;
    const lambda = 20.0; //define the regularization coefficient

    final point1 = [10.0, 20.0, 30.0];
    final point2 = [20.0, 30.0, 40.0];
    final point3 = [70.0, 80.0, 90.0];

    late CoordinateDescentOptimizer optimizer;
    late Matrix data;
    late Matrix labels;

    setUp(() {
      initCommonModule();
      data = Matrix.fromList([point1, point2, point3]);
      labels = Matrix.fromList([
        [2.0],
        [3.0],
        [2.0],
      ]);
      optimizer = CoordinateDescentOptimizer(
        data,
        labels,
        costFunction: const LeastSquareCostFunction(),
        isFittingDataNormalized: true,
        minCoefficientsUpdate: 1e-5,
        iterationsLimit: iterationsNumber,
        initialWeightsType: InitialCoefficientsType.zeroes,
        lambda: lambda,
        dtype: DType.float32,
      );
    });

    tearDown(injector.clearAll);

    /// (The test case explanation)[https://github.com/gyrdym/ml_algo/wiki/Coordinate-descent-optimizer-(regularized-case)-should-find-optimal-weights-for-the-given-data]
    test('should find optimal coefficients for the given data', () {
      // actually, points in this example are not normalized
      final coefficients = optimizer
          .findExtrema()
          .getColumn(0);
      expect(coefficients, equals([-4381770, -4493700, -4073630]));
    });
  });
}
