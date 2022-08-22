import 'dart:math' as math;

import 'package:ml_algo/src/cost_function/least_square_cost_function.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:test/test.dart';

import '../helpers.dart';

void main() {
  group('LeastSquareCostFunction', () {
    final x11 = 21.5;
    final x12 = -0.04;
    final x13 = 27.0;
    final x21 = 15.0;
    final x22 = 0.0;
    final x23 = 33.9;
    final x31 = -10.0;
    final x32 = -1.0;
    final x33 = 22.0;
    final x41 = -10.0;
    final x42 = -1.0;
    final x43 = 17.0;

    final w1 = 0.1;
    final w2 = 0.2;
    final w3 = -0.9;

    final y1 = 11.0;
    final y2 = 17.0;
    final y3 = -25.0;
    final y4 = 5.0;

    final x = Matrix.fromList([
      [x11, x12, x13],
      [x21, x22, x23],
      [x31, x32, x33],
      [x41, x42, x43],
    ]);
    final w = Matrix.column([w1, w2, w3]);
    final y = Matrix.column([y1, y2, y3, y4]);

    final squaredCost = const LeastSquareCostFunction();

    group('LeastSquareCostFunction.getCost', () {
      test('should return residual sum of squares', () {
        final actual = squaredCost.getCost(x, w, y);
        final expected = math.pow(x11 * w1 + x12 * w2 + x13 * w3 - y1, 2) +
            math.pow(x21 * w1 + x22 * w2 + x23 * w3 - y2, 2) +
            math.pow(x31 * w1 + x32 * w2 + x33 * w3 - y3, 2) +
            math.pow(x41 * w1 + x42 * w2 + x43 * w3 - y4, 2);

        expect(actual, closeTo(expected, 1e-3));
      });
    });

    group('LeastSquareCostFunction.getGradient', () {
      test('should return a gradient vector of least square function', () {
        // The formula in matrix notation:
        // -2 * X^t * (y - X*w)
        // where X^t - transposed X matrix
        // y - labels matrix (vector-column)
        // w - coefficients matrix (vector-column)
        final actual = squaredCost.getGradient(x, w, y);
        final expected = [
          [
            -2.0 *
                (x11 * (y1 - (x11 * w1 + x12 * w2 + x13 * w3)) +
                    x21 * (y2 - (x21 * w1 + x22 * w2 + x23 * w3)) +
                    x31 * (y3 - (x31 * w1 + x32 * w2 + x33 * w3)) +
                    x41 * (y4 - (x41 * w1 + x42 * w2 + x43 * w3)))
          ],
          [
            -2.0 *
                (x12 * (y1 - (x11 * w1 + x12 * w2 + x13 * w3)) +
                    x22 * (y2 - (x21 * w1 + x22 * w2 + x23 * w3)) +
                    x32 * (y3 - (x31 * w1 + x32 * w2 + x33 * w3)) +
                    x42 * (y4 - (x41 * w1 + x42 * w2 + x43 * w3)))
          ],
          [
            -2.0 *
                (x13 * (y1 - (x11 * w1 + x12 * w2 + x13 * w3)) +
                    x23 * (y2 - (x21 * w1 + x22 * w2 + x23 * w3)) +
                    x33 * (y3 - (x31 * w1 + x32 * w2 + x33 * w3)) +
                    x43 * (y4 - (x41 * w1 + x42 * w2 + x43 * w3)))
          ],
        ];

        expect(actual, iterable2dAlmostEqualTo(expected, 1e-3));
      });
    });
  });
}
