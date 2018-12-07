import 'dart:typed_data';

import 'package:ml_algo/src/cost_function/cost_function_factory.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:test/test.dart';

void main() {
  group('Squared cost function', () {
    final squaredCost = CostFunctionFactory.squared();

    test('should return a proper gradient vector', () {
      // The formula in matrix notation:
      // -2 * X^t * (y - X*w)
      // where X^t - transposed X matrix
      // y - labels matrix (vector-column)
      // w - coefficients matrix (vector-column)
      final x = Float32x4MatrixFactory.from([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
      ]);
      final w = Float32x4VectorFactory.from([-1.0, 2.0, -3.0]);
      final y = Float32x4VectorFactory.from([10.0, 20.0, 30.0]);
      final expected = [
        [-2 * (16 + 128 + 7 * 48)],
        [-2 * (32 + 160 + 8 * 48)],
        [-2 * (48 + 192 + 9 * 48)],
      ];
      final actual = squaredCost.getGradient(x, w, y);

      expect(actual, equals(expected));
    });
  });

  group('Log likelihood cost function', () {
    final mockedLinkFn = (Float32x4 scores) => Float32x4.splat(1.0);
    final logLikelihoodCost = CostFunctionFactory.logLikelihood(mockedLinkFn);

    test('should return a proper gradient vector', () {
      // The formula in matrix notation:
      // X^t * (indicator(Y, 1) - P(y=+1|X,W))
      // where X^t - transposed X matrix
      // Y - labels matrix (vector-column)
      // W - coefficients matrix (vector-column)
      // indicator - function, that returns 1 if y == 1, otherwise returns 0
      // P(y=+1|X,W) - score to link function
      final x = Float32x4MatrixFactory.from([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
      ]);
      final w = Float32x4VectorFactory.from([-1.0, 2.0, -3.0]);
      final y = Float32x4VectorFactory.from([1.0, 1.0, 0.0]);
      final expected = [
        [-7],
        [-8],
        [-9],
      ];
      final actual = logLikelihoodCost.getGradient(x, w, y);

      expect(actual, equals(expected));
    });
  });
}
