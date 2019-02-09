import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/cost_function/squared.dart';
import 'package:ml_algo/src/optimizer/coordinate/coordinate.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_generator_factory_impl.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../test_utils/mocks.dart';

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
      final initialWeightsGeneratorMock = InitialWeightsGeneratorMock();
      final initialWeightsGeneratorFactoryMock = InitialWeightsGeneratorFactoryMock();
      when(initialWeightsGeneratorFactoryMock.fromType(InitialWeightsType.zeroes))
          .thenReturn(initialWeightsGeneratorMock);

      final costFunctionFactoryMock = CostFunctionFactoryMock();

      optimizer = CoordinateOptimizer(
          initialWeightsGeneratorFactory: initialWeightsGeneratorFactoryMock,
          costFunctionFactory: costFunctionFactoryMock,
          initialWeightsType: InitialWeightsType.zeroes,
          costFunctionType: CostFunctionType.squared,
          minCoefficientsDiff: 1e-5,
          iterationLimit: iterationsNumber,
          lambda: lambda
      );

      data = MLMatrix.from([point1, point2, point3, point4]);
      labels = MLVector.from([20.0, 30.0, 20.0, 40.0]);
    });

    /// Given matrix X:
    /// [10.0, 20.0, 30.0];
    /// [40.0, 50.0, 60.0];
    /// [70.0, 80.0, 90.0];
    /// [20.0, 30.0, 10.0];
    ///
    /// Given labels vector y:
    /// [20.0, 30.0, 20.0, 40.0]
    ///
    /// Put it together:
    ///
    /// [10.0, 20.0, 30.0] [20.0]
    /// [40.0, 50.0, 60.0] [30.0]
    /// [70.0, 80.0, 90.0] [20.0]
    /// [20.0, 30.0, 10.0] [40.0]
    ///
    /// Given lambda: 0.0 (unregularized case)
    ///
    /// Formula for coordinate descent with respect to j column: x_j * (y_i - x_i(-j) * w(-j)),
    ///
    /// where x_j - j-th column (e.g., if j = 0 then x_j = [10.0, 40.0, 70.0, 20.0])
    ///       y_i - i-th label (e.g., if i = 0 then y_i = 20.0)
    ///       x_i(-j) - i-th point, where j coordinate is excluded (e.g.,
    ///         if i = 0 then x_i = [10.0, 20.0, 30.0], if i = 0 and j = 0 then x_i(-j) = [0.0, 20.0, 30.0])
    ///       w(-j) - coefficients vector or weights vector, j term is excluded
    ///
    /// Initial weights:
    /// w = [0.0, 0.0, 0.0]
    ///
    /// -------------------------------------------------------------------------------------------------------------
    /// iteration 1:
    /// j = 0:                         j = 1:                         j = 2:
    /// 10 * (20 - (20 * 0 + 30 * 0))  20 * (20 - (10 * 0 + 30 * 0))  30 * (20 - (10 * 0 + 20 * 0))
    /// 40 * (30 - (50 * 0 + 60 * 0))  50 * (30 - (40 * 0 + 60 * 0))  60 * (30 - (40 * 0 + 50 * 0))
    /// 70 * (20 - (80 * 0 + 90 * 0))  80 * (20 - (70 * 0 + 90 * 0))  90 * (20 - (70 * 0 + 80 * 0))
    /// 20 * (40 - (30 * 0 + 10 * 0))  30 * (40 - (20 * 0 + 10 * 0))  10 * (40 - (20 * 0 + 30 * 0))
    ///
    /// summing up all above (column-wise):
    /// 3600  4700  4600
    ///
    /// weights at the first iteration: w = [3600, 4700, 4600]
    ///
    ///--------------------------------------------------------------------------------------------------------------
    /// iteration 2:
    /// j = 0:                               j = 1:                               j = 2:
    /// 10 * (20 - (20 * 4700 + 30 * 4600))  20 * (20 - (10 * 3600 + 30 * 4600))  30 * (20 - (10 * 3600 + 20 * 4700))
    /// 40 * (30 - (50 * 4700 + 60 * 4600))  50 * (30 - (40 * 3600 + 60 * 4600))  60 * (30 - (40 * 3600 + 50 * 4700))
    /// 70 * (20 - (80 * 4700 + 90 * 4600))  80 * (20 - (70 * 3600 + 90 * 4600))  90 * (20 - (70 * 3600 + 80 * 4700))
    /// 20 * (40 - (30 * 4700 + 10 * 4600))  30 * (40 - (20 * 3600 + 10 * 4600))  10 * (40 - (20 * 3600 + 30 * 4700))
    ///
    /// summing up all above (column-wise):
    /// -81796400 -81295300 -85285400
    ///
    /// weights at the second iteration: w = [-81796400, -81295300, -85285400]
    ///
    /// but we cannot get exactly the same vector as above due to fuzzy arithmetic with floating point numbers. In our case
    /// we will never get exactly -81295300 (second element of the vector w), since 32-bit floating point number has 24 bits
    /// of mantissa precision. 81295300 in binary is 100110110000111011111000100. This requires 25bits of mantissa
    /// precision to store precisely, so the binary number 100 (4 in decimal) will be cut off. Thus we should deposit
    /// some delta for comparision
    ///
    test('should find optimal weights for the given test_data', () {
      final weights = optimizer.findExtrema(data, labels);
      final w1 = weights[0];
      final w2 = weights[1];
      final w3 = weights[2];
      final delta = 5.0;

      expect(w1, closeTo(-81796400, delta));
      expect(w2, closeTo(-81295300, delta));
      expect(w3, closeTo(-85285400, delta));
    }, skip: true);
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
      final initialWeightsGeneratorMock = InitialWeightsGeneratorMock();
      final initialWeightsGeneratorFactoryMock = InitialWeightsGeneratorFactoryMock();
      when(initialWeightsGeneratorFactoryMock.fromType(InitialWeightsType.zeroes))
          .thenReturn(initialWeightsGeneratorMock);

      final costFunctionFactoryMock = CostFunctionFactoryMock();

      optimizer = CoordinateOptimizer(
          initialWeightsGeneratorFactory: initialWeightsGeneratorFactoryMock,
          costFunctionFactory: costFunctionFactoryMock,
          minCoefficientsDiff: 1e-5,
          iterationLimit: iterationsNumber,
          lambda: lambda
      );

      data = MLMatrix.from([point1, point2, point3]);
      labels = MLVector.from([2.0, 3.0, 2.0]);
    });

    /// Given matrix x:
    /// [10.0, 20.0, 30.0]
    /// [20.0, 30.0, 40.0]
    /// [70.0, 80.0, 90.0]
    ///
    /// Given labels vector y:
    /// [2.0]
    /// [3.0]
    /// [2.0]
    ///
    /// Put it together:
    ///
    /// [10.0, 20.0, 30.0] [2.0]
    /// [20.0, 30.0, 40.0] [3.0]
    /// [70.0, 80.0, 90.0] [2.0]
    ///
    /// Lambda: 20.0
    ///
    /// Initial weights: [0.0, 0.0, 0.0]
    ///
    /// Formula for coordinate descent with respect to j column:
    /// x_j * (y_i - x_i(-j) * w(-j)), see explanation above
    ///
    /// Regularization rule:
    /// if w < -lambda / 2 => w = w + lambda / 2
    /// if -lambda / 2 <= w <= lambda / 2 => w = 0 (from subgradient)
    /// if w > lambda / 2 => w = 2 - lambda / 2
    ///
    /// 1st iteration:
    /// j = 0:                                    j = 1:                                    j = 2:
    /// 10.0 * (2.0 - (20.0 * 0.0 + 30.0 * 0.0))  20.0 * (2.0 - (10.0 * 0.0 + 30.0 * 0.0))  30.0 * (2.0 - (10.0 * 0.0 + 20.0 * 0.0))
    /// 20.0 * (3.0 - (30.0 * 0.0 + 40.0 * 0.0))  30.0 * (3.0 - (20.0 * 0.0 + 40.0 * 0.0))  40.0 * (3.0 - (20.0 * 0.0 + 30.0 * 0.0))
    /// 70.0 * (2.0 - (80.0 * 0.0 + 90.0 * 0.0))  80.0 * (2.0 - (70.0 * 0.0 + 90.0 * 0.0))  90.0 * (2.0 - (70.0 * 0.0 + 80.0 * 0.0))
    ///
    /// sum up all above (column-wise):
    /// j = 0:  j = 1:  j = 2:
    /// 20      40      60
    /// 60      90      120
    /// 140     160     180
    /// -------------------
    /// 220     290     360
    ///
    /// unregularized weights after the first iteration: [220, 290, 360]
    /// Let's make them more regular:
    /// w_1 = 220 > 20 / 2 => 220 - 20 / 2 = 210.0
    /// w_2 = 290 > 20 / 2 => 290 - 20 / 2 = 280.0
    /// w_3 = 360 > 20 / 2 => 360 - 20 / 2 = 350.0
    ///
    /// Regularized weights vector after the 1st iteration: [210.0, 280.0, 350.0]
    ///
    /// 2nd iteration:
    /// j = 0:                                        j = 1:                                        j = 2:
    /// 10.0 * (2.0 - (20.0 * 280.0 + 30.0 * 350.0))  20.0 * (2.0 - (10.0 * 210.0 + 30.0 * 350.0))  30.0 * (2.0 - (10.0 * 210.0 + 20.0 * 280.0))
    /// 20.0 * (3.0 - (30.0 * 280.0 + 40.0 * 350.0))  30.0 * (3.0 - (20.0 * 210.0 + 40.0 * 350.0))  40.0 * (3.0 - (20.0 * 210.0 + 30.0 * 280.0))
    /// 70.0 * (2.0 - (80.0 * 280.0 + 90.0 * 350.0))  80.0 * (2.0 - (70.0 * 210.0 + 90.0 * 350.0))  90.0 * (2.0 - (70.0 * 210.0 + 80.0 * 280.0))
    ///
    /// sum up all above:
    /// -160980   -251960   -230940
    /// -447940   -545910   -503880
    /// -3772860  -3695840  -3338820
    /// ----------------------------
    /// -4381780  -4493710  -4073640
    ///
    /// unregularized weights after the first iteration: [-4381780, -4493710, -4073640]
    /// Let's make them more regular:
    /// w_1 = -4381780 < 20 / 2 => -4381780 + 20 / 2 = -4381770.0
    /// w_2 = -4493710 < 20 / 2 => -4493710 + 20 / 2 = -4493700.0
    /// w_3 = -4073640 < 20 / 2 => -4073640 + 20 / 2 = -4073630.0
    ///
    test('should find optimal weights for the given test_data', () {
      // actually, points in this example are not normalized
      final weights = optimizer.findExtrema(data, labels, arePointsNormalized: true);
      final w1 = weights[0];
      final w2 = weights[1];
      final w3 = weights[2];

      expect(w1, equals(-4381770));
      expect(w2, equals(-4493700));
      expect(w3, equals(-4073630));
    }, skip: true);
  });
}
