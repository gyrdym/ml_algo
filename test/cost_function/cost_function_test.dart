import 'package:ml_algo/src/cost_function/log_likelihood.dart';
import 'package:ml_algo/src/cost_function/squared.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../test_utils/mocks.dart';

void main() {
  group('SquaredCost', () {
    final squaredCost = const SquaredCost();

    test('should return a proper gradient vector', () {
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
  });

  group('LogLikelihoodCost', () {
    final mockedLinkFn = LinkFunctionMock();
    final logLikelihoodCost = LogLikelihoodCost(mockedLinkFn);

    when(mockedLinkFn.link(any)).thenReturn(Matrix.fromList([
      [1.0],
      [1.0],
      [1.0],
    ]));

    test('should return a proper gradient vector', () {
      // The formula in matrix notation:
      // X^t * (indicator(Y, 1) - P(y=+1|X,W))
      // where X^t - transposed X matrix
      // Y - labels matrix (vector-column)
      // W - coefficients matrix (vector-column)
      // indicator - function, that returns 1 if y == 1, otherwise returns 0
      // P(y=+1|X,W) - score to link function
      final x = Matrix.fromList([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
      ]);
      // 1 4 7
      // 2 5 8
      // 3 6 9
      final w = Matrix.fromList([
        [-1.0],
        [2.0],
        [-3.0],
      ]);
      final y = Matrix.fromList([
        [1.0],
        [1.0],
        [0.0],
      ]);
      final expected = [
        [-7.0],
        [-8.0],
        [-9.0],
      ];
      final actual = logLikelihoodCost.getGradient(x, w, y);

      expect(actual, equals(expected));
    });
  });
}
