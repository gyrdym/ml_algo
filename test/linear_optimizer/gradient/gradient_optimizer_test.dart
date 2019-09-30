import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../../test_utils/mocks.dart';
import 'gradient_common.dart';

Matrix getPoints() => Matrix.fromList([
      [5.0, 10.0, 15.0],
      [1.0, 2.0, 3.0],
      [10.0, 20.0, 30.0],
      [100.0, 200.0, 300.0],
    ]);

void verifyGetGradientCall(CostFunction mock,
    {Iterable<Iterable<double>> x,
    Iterable<Iterable<double>> w,
    Iterable<Iterable<double>> y,
    int calls}) {
  verify(mock.getGradient(
    argThat(equals(x)),
    argThat(equals(w)),
    argThat(equals(y)),
  )).called(calls);
}

void main() {
  group('Gradient descent solver', () {
    final costFunction = CostFunctionMock();

    test('should properly process `batchSize` parameter when the latter is '
        'equal to `1` (stochastic case)', () {
      final points = getPoints();
      final labels = Matrix.fromList([
        [10.0],
        [20.0],
        [30.0],
        [40.0],
      ]);
      final x = [
        [10.0, 20.0, 30.0]
      ];
      final y = [[30.0]];
      final w1 = [
        [0.0],
        [0.0],
        [0.0],
      ];
      final w2 = [
        [-20.0],
        [-20.0],
        [-20.0],
      ];
      final w3 = [
        [-40.0],
        [-40.0],
        [-40.0],
      ];
      final grad = Matrix.fromList([
        [10.0],
        [10.0],
        [10.0]
      ]);
      final interval = [2, 3];

      mockGetGradient(costFunction, x: x, w: w1, y: y, gradient: grad);
      mockGetGradient(costFunction, x: x, w: w2, y: y, gradient: grad);
      mockGetGradient(costFunction, x: x, w: w3, y: y, gradient: grad);

      when(randomizerMock.getIntegerInterval(0, 4, intervalLength: 1))
          .thenReturn(interval);

      testOptimizer(points, labels, costFunction, (optimizer) {
        optimizer.findExtrema();
        verifyGetGradientCall(costFunction, x: x, w: w1, y: y, calls: 1);
        verifyGetGradientCall(costFunction, x: x, w: w2, y: y, calls: 1);
        verifyGetGradientCall(costFunction, x: x, w: w3, y: y, calls: 1);
      }, iterations: 3, batchSize: 1);
    });

    test('should properly process `batchSize` parameter when the latter is '
        'equal to `2` (mini batch case)', () {
      final points = getPoints();
      final labels = Matrix.fromList([
        [10.0],
        [20.0],
        [30.0],
        [40.0],
      ]);
      final batch = [
        [5.0, 10.0, 15.0],
        [1.0, 2.0, 3.0]
      ];
      final y = [
        [10.0],
        [20.0],
      ];
      final grad = [
        [10.0],
        [10.0],
        [10.0]
      ];

      when(randomizerMock.getIntegerInterval(0, 4, intervalLength: 2))
          .thenReturn([0, 2]);

      when(costFunction.getGradient(
              argThat(equals(batch)), any, argThat(equals(y))))
          .thenReturn(Matrix.fromList(grad));

      testOptimizer(points, labels, costFunction, (optimizer) {
        optimizer.findExtrema();
        verify(costFunction.getGradient(
                argThat(equals(batch)), any, argThat(equals(y))))
            .called(3); // 3 iterations
      }, iterations: 3, batchSize: 2);
    });

    test('should properly process `batchSize` parameter when the latter is '
        'equal to `4` (batch case)', () {
      final points = getPoints();
      final labels = Matrix.fromList([
        [10.0],
        [20.0],
        [30.0],
        [40.0],
      ]);

      when(randomizerMock.getIntegerInterval(0, 4, intervalLength: 4))
          .thenReturn([0, 4]);
      when(costFunction.getGradient(
              argThat(equals(points)), any, argThat(equals(labels))))
          .thenReturn(Matrix.fromList([
            [10.0],
            [10.0],
            [10.0]
      ]));

      testOptimizer(points, labels, costFunction, (optimizer) {
        optimizer.findExtrema();
        verify(costFunction.getGradient(
                argThat(equals(points)), any, argThat(equals(labels))))
            .called(3); // 3 iterations
      }, iterations: 3, batchSize: 4);
    });

    test('should cut the batch size parameter to make it equal to the number '
        'of given points', () {
      final iterationLimit = 3;
      final points = getPoints();
      final labels = Matrix.fromList([
        [10.0],
        [20.0],
        [30.0],
        [40.0],
      ]);
      final interval = [0, 4];
      final grad = [
        [10.0],
        [10.0],
        [10.0],
      ];

      when(randomizerMock.getIntegerInterval(0, 4, intervalLength: 4))
          .thenReturn(interval);
      when(costFunction.getGradient(any, any, any))
          .thenReturn(Matrix.fromList(grad));

      testOptimizer(points, labels, costFunction, (optimizer) {
        optimizer.findExtrema();
        verify(costFunction.getGradient(
                argThat(equals(points)), any, argThat(equals(labels))))
            .called(iterationLimit);
        verify(learningRateGeneratorMock.getNextValue()).called(iterationLimit);
      }, batchSize: 4, iterations: iterationLimit);
    });

    /// (Explanation of the test case)[https://github.com/gyrdym/ml_algo/wiki/Gradient-descent-optimizer-should-find-optimal-coefficient-values]
    test('should find optimal coefficient values', () {
      final points = Matrix.fromList([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
      ]);
      final labels = Matrix.fromList([
        [7.0],
        [8.0],
      ]);

      when(randomizerMock.getIntegerInterval(0, 2, intervalLength: 2))
          .thenReturn([0, 2]);
      when(costFunction.getGradient(
              argThat(equals(points)), any, argThat(equals(labels))))
          .thenReturn(Matrix.fromList([
            [8.0],
            [8.0],
            [8.0]
      ]));

      testOptimizer(points, labels, costFunction, (optimizer) {
        final optimalCoefficients = optimizer.findExtrema();
        expect(optimalCoefficients, equals([
          [-48.0],
          [-48.0],
          [-48.0]
        ]));
        expect(optimalCoefficients.columnsNum, 1);
        expect(optimalCoefficients.rowsNum, 3);
      }, iterations: 3, batchSize: 2);
    });

    /// (Explanation of the test case)[https://github.com/gyrdym/ml_algo/wiki/Gradient-descent-optimizer-should-find-optimal-coefficient-values-and-regularize-it]
    test('should find optimal coefficient values and regularize it', () {
      final points = Matrix.fromList([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
      ]);
      final labels = Matrix.fromList([
        [7.0],
        [8.0]
      ]);
      final gradient = Matrix.fromList([
        [8.0],
        [8.0],
        [8.0],
      ]);

      when(randomizerMock.getIntegerInterval(0, 2, intervalLength: 2))
          .thenReturn([0, 2]);
      when(costFunction.getGradient(
              argThat(equals(points)), any, argThat(equals(labels))))
          .thenReturn(gradient);

      testOptimizer(points, labels, costFunction, (optimizer) {
        final optimalCoefficients = optimizer.findExtrema();
        expect(optimalCoefficients, equals([
          [-23728.0],
          [-23728.0],
          [-23728.0]
        ]));
        expect(optimalCoefficients.columnsNum, 1);
        expect(optimalCoefficients.rowsNum, 3);
      }, iterations: 3, batchSize: 2, lambda: 10.0);
    });

    test('should consider `learningRate` parameter', () {
      final initialLearningRate = 10.0;
      final iterations = 3;
      testOptimizer(Matrix.fromList([[]]), Matrix.fromList([[]]), costFunction,
        (optimizer) {
          verify(learningRateGeneratorMock.init(initialLearningRate)).called(1);
        },
        iterations: iterations,
        batchSize: 1,
        lambda: 0.0,
        eta: initialLearningRate,
        verifyConvergenceDetectorCall: false);
    });
  });
}
