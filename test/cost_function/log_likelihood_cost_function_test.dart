import 'dart:math' as math;

import 'package:ml_algo/src/cost_function/log_likelihood_cost_function.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../mocks.mocks.dart';

void main() {
  group('LogLikelihoodCostFunction', () {
    final mockedLinkFn = MockLinkFunction();
    final positiveLabel = 10.0;
    final negativeLabel = -10.0;
    final logLikelihoodCost = LogLikelihoodCostFunction(
        mockedLinkFn, positiveLabel, negativeLabel);
    final x = Matrix.fromList([
      [1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0],
      [7.0, 8.0, 9.0],
    ]);
    final w = Matrix.column([
      -1.0,
       2.0,
      -3.0,
    ]);
    final y = Matrix.column([
      positiveLabel,
      positiveLabel,
      negativeLabel,
    ]);

    setUp(() {
      when(
        mockedLinkFn.link(any),
      ).thenReturn(
        Matrix.column([1, 1, 1]),
      );
    });

    tearDown(resetMockitoState);

    test('should return a gradient vector of log likelihood error '
        'function', () {
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
        () => logLikelihoodCost.getSubGradient(
            1,
            Matrix.empty(),
            Matrix.empty(),
            Matrix.empty(),
        ),
        throwsUnimplementedError,
      );
    });

    test('should return log likelihood cost', () {
      reset(mockedLinkFn);
      when(
        mockedLinkFn.link(any),
      ).thenReturn(
        Matrix.column([0.3, 0.7, 0.2]),
      );

      final cost = logLikelihoodCost.getCost(x, w, y);

      verify(
          mockedLinkFn.link(
              argThat(
                  equals(
                      [
                        [-6],
                        [-12],
                        [-18],
                      ],
                  ),
              ),
          ),
      );
      expect(cost, closeTo(math.log(0.3) + math.log(0.7) + math.log(0.8), 1e-4));
    });
  });
}
