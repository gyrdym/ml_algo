import 'package:ml_algo/src/common/exception/matrix_column_exception.dart';
import 'package:ml_algo/src/cost_function/log_likelihood_cost_function.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../mocks.dart';

void main() {
  group('LogLikelihoodCostFunction', () {
    final mockedLinkFn = LinkFunctionMock();
    final logLikelihoodCost = LogLikelihoodCostFunction(mockedLinkFn);

    when(mockedLinkFn.link(any)).thenReturn(Matrix.column([1, 1, 1]));

    test('should return a gradient vector of log likelihood error '
        'function', () {
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

    test('should return a subgradient vector', () {
      expect(
        () => logLikelihoodCost.getSubGradient(null, null, null, null),
        throwsUnimplementedError,
      );
    });

    group('LogLikelihoodCostFunction.getCost', () {
      test('should throw an exception if predictedLabels are not a matrix '
          'column', () {
        final predictedLabels = Matrix.fromList([
          [ 1,   2],
          [-1, 200],
          [-1, 400],
        ]);
        final originalLabels = Matrix.column([11, 17, -30, 10, 9, 0]);
        final actual = () => logLikelihoodCost.getCost(predictedLabels, originalLabels);

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
        final actual = () => logLikelihoodCost.getCost(predictedLabels, originalLabels);

        expect(actual, throwsA(isA<MatrixColumnException>()));
      });
    });
  });
}
