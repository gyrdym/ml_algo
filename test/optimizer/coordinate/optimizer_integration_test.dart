import 'dart:typed_data';

import 'package:dart_ml/src/core/implementation.dart';
import 'package:dart_ml/src/core/interface.dart';
import 'package:dart_ml/src/di/injector.dart' show coreInjector;
import 'package:di/di.dart';
import 'package:simd_vector/vector.dart';
import 'package:test/test.dart';

void main() {
  group('Coordinate descent optimizer', () {
    const iterationsNumber = 2;
    const lambda = 0.0;

    final point1 = new Float32x4Vector.from([10.0, 20.0, 30.0]);
    final point2 = new Float32x4Vector.from([40.0, 50.0, 60.0]);
    final point3 = new Float32x4Vector.from([70.0, 80.0, 90.0]);
    final point4 = new Float32x4Vector.from([20.0, 30.0, 10.0]);

    Optimizer optimizer;
    List<Float32x4Vector> data;
    Float32List labels;

    setUp(() {
      coreInjector = new ModuleInjector([
        new Module()
          ..bind(InitialWeightsGenerator, toValue: InitialWeightsGeneratorFactory.createZeroWeightsGenerator())
          ..bind(ScoreFunction, toValue: ScoreFunctionFactory.Linear())
      ]);

      optimizer = CoordinateOptimizerFactory.createCoordinateOptimizerFactory(1e-5, iterationsNumber, lambda);

      data = [point1, point2, point3, point4];
      labels = new Float32List.fromList([20.0, 30.0, 20.0, 40.0]);
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
    /// Write it together:
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
    /// where x_j - column with index j (e.g., if j = 0 then x_j = [10.0, 40.0, 70.0, 20.0])
    ///       y_i - label on i-th row (e.g., if i = 0 then y_i = 20.0)
    ///       x_i(-j) - point on i-th row, data vector, j coordinate is excluded (e.g.,
    ///       if i = 0 then x_i(-j) = [10.0, 20.0, 30.0], if i = 0 and j = 0 then x_i(-j) = [20.0, 30.0])
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
    /// summing all above up (column-wise):
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
    /// summing all above up (column-wise):
    /// -81796400 -81295300 -85285400
    ///
    /// weights at the second iteration: w = [-81796400, -81295300, -85285400]
    ///
    /// but we cannot get exactly the same vector as above due to fuzzy arithmetic with floating point numbers. In our case
    /// we will never get exactly -81295300 (second element of the vector w), since 32-bit floating point number has 24 bits
    /// of mantissa precision. 81295300 in binary is 100110110000111011111000100. This requires 25bits of mantissa
    /// precision to store precisely, so the number 100 (4 in decimal) will be cut off. Thus we should deposit some delta
    /// to comparision
    ///
    test('should find optimal weights for the given data', () {
      final weights = optimizer.findExtrema(data, labels).asList();
      final w1 = weights[0];
      final w2 = weights[1];
      final w3 = weights[2];
      final delta = 5.0;

      expect(w1, closeTo(-81796400, delta));
      expect(w2, closeTo(-81295300, delta));
      expect(w3, closeTo(-85285400, delta));
    });
  });
}
