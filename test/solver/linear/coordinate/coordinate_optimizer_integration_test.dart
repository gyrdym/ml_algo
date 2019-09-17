import 'package:ml_algo/src/cost_function/squared.dart';
import 'package:ml_algo/src/solver/linear/coordinate/coordinate.dart';
import 'package:ml_algo/src/solver/linear/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:ml_tech/unit_testing/matchers/iterable_almost_equal_to.dart';
import 'package:test/test.dart';

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

    CoordinateOptimizer optimizer;
    Matrix data;
    Matrix labels;

    setUp(() {
      data = Matrix.fromList([point1, point2, point3, point4]);
      labels = Matrix.fromList([
        [20.0],
        [30.0],
        [20.0],
        [40.0]
      ]);
      optimizer = CoordinateOptimizer(
          data, labels,
          initialWeightsType: InitialWeightsType.zeroes,
          costFunction: const SquaredCost(),
          minCoefficientsDiff: 1e-5,
          iterationsLimit: iterationsNumber,
          lambda: lambda);
    });

    /// (The test case explanation)[https://github.com/gyrdym/ml_algo/wiki/Coordinate-descent-optimizer-(unregularized-case)-should-find-optimal-weights-for-the-given-data]
    test('should find optimal weights for the given data', () {
      final weights = optimizer.findExtrema();
      final expected = [-81796400.0, -81295300.0, -85285400.0];
      expect(weights.rowsNum, 1);
      expect(weights.columnsNum, 3);
      expect(weights.getRow(0), iterableAlmostEqualTo(expected, 5.0));
    });
  });

  group('Coordinate descent solver (regularized case)', () {
    const iterationsNumber = 2;
    const lambda = 20.0; //define the regularization coefficient

    final point1 = [10.0, 20.0, 30.0];
    final point2 = [20.0, 30.0, 40.0];
    final point3 = [70.0, 80.0, 90.0];

    CoordinateOptimizer optimizer;
    Matrix data;
    Matrix labels;

    setUp(() {
      data = Matrix.fromList([point1, point2, point3]);
      labels = Matrix.fromList([
        [2.0],
        [3.0],
        [2.0],
      ]);
      optimizer = CoordinateOptimizer(
          data, labels,
          costFunction: const SquaredCost(),
          isTrainDataNormalized: true,
          minCoefficientsDiff: 1e-5,
          iterationsLimit: iterationsNumber,
          initialWeightsType: InitialWeightsType.zeroes,
          lambda: lambda);
    });

    /// (The test case explanation)[https://github.com/gyrdym/ml_algo/wiki/Coordinate-descent-optimizer-(regularized-case)-should-find-optimal-weights-for-the-given-data]
    test('should find optimal weights for the given data', () {
      // actually, points in this example are not normalized
      final weights = optimizer
          .findExtrema()
          .getRow(0);
      expect(weights, equals([-4381770, -4493700, -4073630]));
    });
  });
}
