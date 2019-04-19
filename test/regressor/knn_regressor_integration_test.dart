import 'package:ml_algo/ml_algo.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:test/test.dart';

void main() {
  group('KNNRegressor (integration)', () {
    test('should predict values', () {
      final k = 2;
      final features = Matrix.from([
        [20, 20, 20, 20, 20],
        [30, 30, 30, 30, 30],
        [15, 15, 15, 15, 15],
        [25, 25, 25, 25, 25],
        [10, 10, 10, 10, 10],
      ]);
      final outcomes = Matrix.from([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
      ]);
      final testFeatures = Matrix.from([
        [9.0, 9.0, 9.0, 9.0, 9.0],
      ]);
      final regressor = NoNParametricRegressor.nearestNeighbor(k: k)
        ..fit(features, outcomes);

      final actual = regressor.predict(testFeatures);
      expect(actual, equals([[4.0]]));
    });
  });
}
