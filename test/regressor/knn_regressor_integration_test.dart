import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:test/test.dart';

void main() {
  group('KNNRegressor (integration)', () {
    test('should predict values using uniform kernel', () {
      final k = 2;
      final data = DataFrame(<Iterable<num>>[
        [20, 20, 20, 20, 20, 1],
        [30, 30, 30, 30, 30, 2],
        [15, 15, 15, 15, 15, 3],
        [25, 25, 25, 25, 25, 4],
        [10, 10, 10, 10, 10, 5],
      ],
          header: ['first', 'second', 'third', 'fourth', 'fifth', 'target'],
          headerExists: false);

      final testFeatures = Matrix.fromList([
        [9.0, 9.0, 9.0, 9.0, 9.0],
      ]);

      final regressor = ParameterlessRegressor.knn(data, 'target', k: k);

      final actual = regressor.predict(testFeatures);
      expect(actual, equals([[4.0]]));
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

      final regressor = ParameterlessRegressor.knn(data, 'target',
        k: k,
        kernel: Kernel.epanechnikov,
      );

      final actual = regressor.predict(testFeatures);
      expect(actual, equals([[-208.875]]));
    });
  });
}
