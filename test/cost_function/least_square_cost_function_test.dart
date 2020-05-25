import 'package:ml_algo/src/common/exception/matrix_column_exception.dart';
import 'package:ml_algo/src/cost_function/least_square_cost_function.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:test/test.dart';

void main() {
  group('LeastSquareCostFunction', () {
    final squaredCost = const LeastSquareCostFunction();

    test('should return a gradient vector of least square function', () {
      // The formula in matrix notation:
      // -2 * X^t * (y - X*w)
      // where X^t - transposed X matrix
      // y - labels matrix (vector-column)
      // w - coefficients matrix (vector-column)
      final x = Matrix.fromList([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
      ]);
      final w = Matrix.fromList([
        [-1.0],
        [2.0],
        [-3.0],
      ]);
      final y = Matrix.fromList([
        [10.0],
        [20.0],
        [30.0]
      ]);
      final expected = [
        [-2.0 * (16 + 128 + 7 * 48)],
        [-2.0 * (32 + 160 + 8 * 48)],
        [-2.0 * (48 + 192 + 9 * 48)],
      ];
      final actual = squaredCost.getGradient(x, w, y);

      expect(actual, equals(expected));
    });

    test('should return residual sum of squares', () {
      final predictedLabels = Matrix.column([10, 20, 30, 40, 50, 6]);
      final originalLabels = Matrix.column([11, 17, -30, 10, 9, 0]);
      final actual = squaredCost.getCost(predictedLabels, originalLabels);
      final expected = 1 + 9 + 3600 + 900 + 41 * 41 + 36;

      expect(actual, expected);
    });

    test('should throw an exception if predictedLabels are not a matrix '
        'column', () {
      final predictedLabels = Matrix.fromList([
        [ 1,   2],
        [-1, 200],
        [-1, 400],
      ]);
      final originalLabels = Matrix.column([11, 17, -30, 10, 9, 0]);
      final actual = () => squaredCost.getCost(predictedLabels, originalLabels);

      expect(actual, throwsA(isA<MatrixColumnException>()));
    });

    test('should throw an exception if originalLabels are not a matrix '
        'column', () {
      final predictedLabels = Matrix.column([10, 20, 30, 40, 50, 6]);
      final originalLabels = Matrix.fromList([
        [1,  -1],
        [3, 100],
        [2,   5],
      ]);
      final actual = () => squaredCost.getCost(predictedLabels, originalLabels);

      expect(actual, throwsA(isA<MatrixColumnException>()));
    });
  });
}
