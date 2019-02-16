import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/optimizer/coordinate/coordinate.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:test/test.dart';

import '../../test_utils/helpers/floating_point_iterable_matchers.dart';

/// L1 regularization, as known as Lasso, is aimed to penalize unimportant features, setting their weights to the zero,
/// therefore, we can treat the objective of the Lasso Optimizer like feature selection. Since lasso optimizer regularizes
/// coefficients by adding to their magnitude their L1-norm, we cannot use gradient methods any longer. Instead, we can
/// use coordinate descent optimization.

void main() {
  group('Coordinate descent optimizer (unregularized case)', () {
    const iterationsNumber = 2;
    const lambda = 0.0;

    final point1 = [10.0, 20.0, 30.0];
    final point2 = [40.0, 50.0, 60.0];
    final point3 = [70.0, 80.0, 90.0];
    final point4 = [20.0, 30.0, 10.0];

    CoordinateOptimizer optimizer;
    MLMatrix data;
    MLVector labels;

    setUp(() {
      optimizer = CoordinateOptimizer(
          initialWeightsType: InitialWeightsType.zeroes,
          costFunctionType: CostFunctionType.squared,
          minCoefficientsDiff: 1e-5,
          iterationsLimit: iterationsNumber,
          lambda: lambda);

      data = MLMatrix.from([point1, point2, point3, point4]);
      labels = MLVector.from([20.0, 30.0, 20.0, 40.0]);
    });

    /// (The test case explanation)[https://github.com/gyrdym/ml_algo/wiki/Coordinate-descent-optimizer-(unregularized-case)-should-find-optimal-weights-for-the-given-data]
    test('should find optimal weights for the given data', () {
      final weights = optimizer.findExtrema(data, labels).getRow(0);
      final expected = [-81796400.0, -81295300.0, -85285400.0];
      expect(weights, vectorAlmostEqualTo(expected, 5.0));
    });
  });

  group('Coordinate descent optimizer (regularized case)', () {
    const iterationsNumber = 2;
    const lambda = 20.0; //define the regularization coefficient

    final point1 = [10.0, 20.0, 30.0];
    final point2 = [20.0, 30.0, 40.0];
    final point3 = [70.0, 80.0, 90.0];

    CoordinateOptimizer optimizer;
    MLMatrix data;
    MLVector labels;

    setUp(() {
      optimizer = CoordinateOptimizer(
          minCoefficientsDiff: 1e-5,
          iterationsLimit: iterationsNumber,
          initialWeightsType: InitialWeightsType.zeroes,
          lambda: lambda);

      data = MLMatrix.from([point1, point2, point3]);
      labels = MLVector.from([2.0, 3.0, 2.0]);
    });

    /// (The test case explanation)[https://github.com/gyrdym/ml_algo/wiki/Coordinate-descent-optimizer-(regularized-case)-should-find-optimal-weights-for-the-given-data]
    test('should find optimal weights for the given data', () {
      // actually, points in this example are not normalized
      final weights = optimizer
          .findExtrema(data, labels, arePointsNormalized: true)
          .getRow(0);
      expect(weights, equals([-4381770, -4493700, -4073630]));
    });
  });
}
