import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:test/test.dart';

void main() {
  group('KnnRegressor', () {
    test('should predict values using uniform kernel', () {
      final k = 2;
      final data = DataFrame(
        <Iterable<num>>[
          [20, 20, 20, 20, 20, 1],
          [30, 30, 30, 30, 30, 2],
          [15, 15, 15, 15, 15, 3],
          [25, 25, 25, 25, 25, 4],
          [10, 10, 10, 10, 10, 5],
        ],
        header: ['first', 'second', 'third', 'fourth', 'fifth', 'target'],
        headerExists: false,
      );

      final testFeatures = Matrix.fromList([
        [9.0, 9.0, 9.0, 9.0, 9.0],
      ]);

      final targetName = 'target';

      final regressor = KnnRegressor(
        data,
        targetName,
        k: k,
        kernel: Kernel.uniform,
        distance: Distance.euclidean,
      );

      final actual = regressor.predict(
        DataFrame.fromMatrix(testFeatures),
      );

      expect(actual.header, equals(['target']));
      expect(actual.toMatrix(), equals([[4.0]]));
    });

    test('should predict values using epanechnikov kernel', () {
      final k = 2;
      final data = DataFrame(<Iterable<num>>[
        [20, 20, 20, 20, 20, 1],
        [30, 30, 30, 30, 30, 2],
        [15, 15, 15, 15, 15, 3],
        [25, 25, 25, 25, 25, 4],
        [10, 10, 10, 10, 10, 5],
      ],
          header: ['first', 'second', 'third', 'fourth', 'fifth', 'target'],
          headerExists: false
      );

      final testFeatures = Matrix.fromList([
        [9.0, 9.0, 9.0, 9.0, 9.0],
      ]);

      final regressor = KnnRegressor(data, 'target',
        k: k,
        kernel: Kernel.gaussian,
      );

      final actual = regressor.predict(
        DataFrame.fromMatrix(testFeatures),
      );

      expect(actual.header, equals(['target']));
      expect(actual.toMatrix(), equals([[5.0]]));
    });
  });
}
