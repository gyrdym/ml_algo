import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import 'gradient_common.dart';

MLMatrix getPoints() => MLMatrix.from([
      [5.0, 10.0, 15.0],
      [1.0, 2.0, 3.0],
      [10.0, 20.0, 30.0],
      [100.0, 200.0, 300.0],
    ]);

void verifyNeverGetGradientCall(
    {Iterable<Iterable<double>> x, Iterable<double> w, Iterable<double> y}) {
  verifyNever(costFunctionMock.getGradient(
    argThat(equals(x)),
    argThat(equals(w)),
    argThat(equals(y)),
  ));
}

void verifyGetGradientCall(CostFunction mock,
    {Iterable<Iterable<double>> x,
    Iterable<double> w,
    Iterable<double> y,
    int calls}) {
  verify(mock.getGradient(
    argThat(equals(x)),
    argThat(equals(w)),
    argThat(equals(y)),
  )).called(calls);
}

void main() {
  group('Gradient descent optimizer', () {
    tearDown(() {
      verify(convergenceDetectorMock.isConverged(any, any)).called(4);
      resetMockitoState();
    });

    test('should properly process `batchSize` parameter when '
        'the latter is equal to `1` (stochastic case)', () {
      final points = getPoints();
      final labels = MLVector.from([10.0, 20.0, 30.0, 40.0]);
      final x = [[10.0, 20.0, 30.0]];
      final y = [30.0];
      final w1 = [0.0, 0.0, 0.0];
      final w2 = [-20.0, -20.0, -20.0];
      final w3 = [-40.0, -40.0, -40.0];
      final grad = [10.0, 10.0, 10.0];
      final interval = [2, 3];

      mockGetGradient(costFunctionMock, x: x, w: w1, y: y, gradient: grad);
      mockGetGradient(costFunctionMock, x: x, w: w2, y: y, gradient: grad);
      mockGetGradient(costFunctionMock, x: x, w: w3, y: y, gradient: grad);

      when(randomizerMock.getIntegerInterval(0, 4, intervalLength: 1))
          .thenReturn(interval);

      testOptimizer((optimizer) {
        optimizer.findExtrema(points, labels);

        verifyGetGradientCall(costFunctionMock, x: x, w: w1, y: y, calls: 1);
        verifyGetGradientCall(costFunctionMock, x: x, w: w2, y: y, calls: 1);
        verifyGetGradientCall(costFunctionMock, x: x, w: w3, y: y, calls: 1);
      }, iterations: 3, batchSize: 1);
    });

    test('should properly process `batchSize` parameter when the latter is '
        'equal to `2` (mini batch case)', () {
      final points = getPoints();
      final labels = MLVector.from([10.0, 20.0, 30.0, 40.0]);
      final batch = [
        [5.0, 10.0, 15.0],
        [1.0, 2.0, 3.0]
      ];
      final y = [10.0, 20.0];
      final grad = [10.0, 10.0, 10.0];

      when(randomizerMock.getIntegerInterval(0, 4, intervalLength: 2))
          .thenReturn([0, 2]);

      when(costFunctionMock.getGradient(
          argThat(equals(batch)),
          any,
          argThat(equals(y))))
          .thenReturn(MLVector.from(grad));

      testOptimizer((optimizer) {
        optimizer.findExtrema(points, labels);

        verify(costFunctionMock.getGradient(
            argThat(equals(batch)),
            any,
            argThat(equals(y)))).called(3); // 3 iterations
      }, iterations: 3, batchSize: 2);
    });

    test('should properly process `batchSize` parameter when the latter is '
        'equal to `4` (batch case)', () {
      final points = getPoints();
      final labels = MLVector.from([10.0, 20.0, 30.0, 40.0]);

      when(randomizerMock.getIntegerInterval(0, 4, intervalLength: 4))
          .thenReturn([0, 4]);
      when(costFunctionMock.getGradient(
              argThat(equals(points)), any, argThat(equals(labels))))
          .thenReturn(MLVector.from([10.0, 10.0, 10.0]));

      testOptimizer((optimizer) {
        optimizer.findExtrema(points, labels);
        verify(costFunctionMock.getGradient(
            argThat(equals(points)), any, argThat(equals(labels))))
            .called(3); // 3 iterations
      }, iterations: 3, batchSize: 4);
    });

    test('should cut the batch size parameter to make it equal to the number '
        'of given points', () {
      final iterationLimit = 3;
      final points = getPoints();
      final labels = MLVector.from([10.0, 20.0, 30.0, 40.0]);
      final interval = [0, 4];
      final grad = [10.0, 10.0, 10.0];

      when(randomizerMock.getIntegerInterval(0, 4, intervalLength: 4))
          .thenReturn(interval);
      when(costFunctionMock.getGradient(any, any, any))
          .thenReturn(MLVector.from(grad));

      testOptimizer((optimizer) {
        optimizer.findExtrema(points, labels);
        verify(costFunctionMock.getGradient(
            argThat(equals(points)), any, argThat(equals(labels))))
            .called(iterationLimit);
        verify(learningRateGeneratorMock.getNextValue()).called(iterationLimit);
      }, batchSize: 15, iterations: iterationLimit);
    });

    /// (Explanation of the test case)[https://github.com/gyrdym/ml_algo/wiki/Gradient-descent-optimizer-should-find-optimal-coefficient-values]
    test('should find optimal coefficient values', () {
      final points = MLMatrix.from([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
      ]);
      final labels = MLVector.from([7.0, 8.0]);

      when(randomizerMock.getIntegerInterval(0, 2, intervalLength: 2))
          .thenReturn([0, 2]);
      when(costFunctionMock.getGradient(
              argThat(equals(points)), any, argThat(equals(labels))))
          .thenReturn(MLVector.from([8.0, 8.0, 8.0]));

      testOptimizer((optimizer) {
        final optimalCoefficients = optimizer.findExtrema(points, labels);
        expect(optimalCoefficients.getRow(0).toList(),
            equals([-48.0, -48.0, -48.0]));
        expect(optimalCoefficients.rowsNum, 1);
      }, iterations: 3, batchSize: 2);
    });

    /// (Explanation of the test case)[https://github.com/gyrdym/ml_algo/wiki/Gradient-descent-optimizer-should-find-optimal-coefficient-values-and-regularize-it]
    test('should find optimal coefficient values and regularize it', () {
      final points = MLMatrix.from([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
      ]);
      final labels = MLVector.from([7.0, 8.0]);
      final gradient = MLVector.from([8.0, 8.0, 8.0]);

      when(randomizerMock.getIntegerInterval(0, 2, intervalLength: 2))
          .thenReturn([0, 2]);
      when(costFunctionMock.getGradient(
              argThat(equals(points)), any, argThat(equals(labels))))
          .thenReturn(gradient);

      testOptimizer((optimizer) {
        final optimalCoefficients = optimizer.findExtrema(points, labels);
        expect(optimalCoefficients.getRow(0).toList(),
            equals([-23728.0, -23728.0, -23728.0]));
        expect(optimalCoefficients.rowsNum, 1);
      }, iterations: 3, batchSize: 2, lambda: 10.0);
    });

//    test('should consider `minCoefficientsUpdate` parameter', () {
//      final minCoefficientsUpdate = 4.0;
//      final points = MLMatrix.from([
//        [1.0, 2.0, 3.0],
//      ]);
//      final labels = MLVector.from([1.0]);
//      final optimizer = createOptimizer(
//          minCoeffUpdate: minCoefficientsUpdate,
//          iterationsLimit: 1000000,
//          batchSize: 1,
//          lambda: 0.0);
//
//      when(randomizerMock.getIntegerInterval(0, 1, intervalLength: 1))
//          .thenReturn([0, 1]);
//      when(costFunctionMock.getGradient(
//        argThat(equals(points)),
//        argThat(equals([0.0, 0.0, 0.0])),
//        argThat(equals(labels)),
//      )).thenReturn(MLVector.from([5.0, 5.0, 5.0]));
//
//      when(costFunctionMock.getGradient(
//        argThat(equals(points)),
//        argThat(equals([-10.0, -10.0, -10.0])),
//        argThat(equals(labels)),
//      )).thenReturn(MLVector.from([2.0, 2.0, 2.0]));
//
//      when(costFunctionMock.getGradient(
//        argThat(equals(points)),
//        argThat(equals([-14.0, -14.0, -14.0])),
//        argThat(equals(labels)),
//      )).thenReturn(MLVector.from([1.0, 1.0, 1.0]));
//
//      // c_1 = c_1_prev - eta * partial = 0 - 2 * 5 = -10
//      // c_2 = c_2_prev - eta * partial = 0 - 2 * 5 = -10
//      // c_3 = c_3_prev - eta * partial = 0 - 2 * 5 = -10
//      //
//      // c = [-10, -10, -10]
//      // distance = sqrt((0 - (-10))^2 + (0 - (-10))^2 + (0 - (-10))^2) = sqrt(300) ~~ 17.32
//      //
//      // c_1 = c_1_prev - eta * partial = -10 - 2 * 2 = -14
//      // c_2 = c_2_prev - eta * partial = -10 - 2 * 2 = -14
//      // c_3 = c_3_prev - eta * partial = -10 - 2 * 2 = -14
//      //
//      // c = [-14, -14, -14]
//      // distance = sqrt((-10 - (-14))^2 + (-10 - (-14))^2 + (-10 - (-14))^2) = sqrt(48) ~~ 6.92
//      //
//      // c_1 = c_1_prev - eta * partial = -14 - 2 * 1 = -16
//      // c_2 = c_2_prev - eta * partial = -14 - 2 * 1 = -16
//      // c_3 = c_3_prev - eta * partial = -14 - 2 * 1 = -16
//      //
//      // c = [-16, -16, -16]
//      // distance = sqrt((-14 - (-16))^2 + (-14 - (-16))^2 + (-14 - (-16))^2) = sqrt(12) ~~ 3.46
//      final coefficients = optimizer.findExtrema(points, labels);
//      verify(learningRateGeneratorMock.getNextValue()).called(3);
//      expect(coefficients.getRow(0).toList(), equals([-16.0, -16.0, -16.0]));
//      expect(coefficients.rowsNum, 1);
//    });
//
//    test('should consider `learningRate` parameter', () {
//      final initialLearningRate = 10.0;
//      createOptimizer(
//          minCoeffUpdate: 1e-100,
//          iterationsLimit: 3,
//          batchSize: 1,
//          lambda: 0.0,
//          eta: initialLearningRate);
//      verify(learningRateGeneratorMock.init(initialLearningRate)).called(1);
//    }, skip: true);
  });
}
