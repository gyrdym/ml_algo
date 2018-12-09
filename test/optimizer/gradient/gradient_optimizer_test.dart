import 'dart:typed_data';

import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/math/randomizer/randomizer.dart';
import 'package:ml_algo/src/optimizer/gradient.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_generator.dart';
import 'package:ml_algo/src/optimizer/learning_rate_generator/generator.dart';
import 'package:mockito/mockito.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:test/test.dart';

class RandomizerMock extends Mock implements Randomizer {}

class InitialWeightsGeneratorMock extends Mock implements InitialWeightsGenerator<Float32x4> {}

class LearningRateGeneratorMock extends Mock implements LearningRateGenerator {}

class CostFunctionMock extends Mock implements CostFunction<Float32x4> {}

Randomizer randomizerMock;
LearningRateGenerator learningRateGeneratorMock;
CostFunction<Float32x4> costFunctionMock;
InitialWeightsGenerator<Float32x4> initialWeightsGeneratorMock;

LearningRateGenerator createLearningRateGenerator() {
  final mock = LearningRateGeneratorMock();
  when(mock.getNextValue()).thenReturn(2.0);
  return mock;
}

InitialWeightsGenerator<Float32x4> createInitialWeightsGenerator() {
  final mock = InitialWeightsGeneratorMock();
  when(mock.generate(3)).thenReturn(Float32x4VectorFactory.from([0.0, 0.0, 0.0]));
  return mock;
}

MLMatrix<Float32x4> getPoints() => Float32x4MatrixFactory.from([
      [5.0, 10.0, 15.0],
      [1.0, 2.0, 3.0],
      [10.0, 20.0, 30.0],
      [100.0, 200.0, 300.0],
    ]);

GradientOptimizer createOptimizer(
        {double eta, double minCoeffUpdate, int iterationsLimit, double lambda, int batchSize}) =>
    GradientOptimizer(randomizerMock, costFunctionMock, learningRateGeneratorMock, initialWeightsGeneratorMock,
        initialLearningRate: eta,
        minCoefficientsUpdate: minCoeffUpdate,
        iterationLimit: iterationsLimit,
        lambda: lambda,
        batchSize: batchSize);

void mockGetGradient({Iterable<Iterable<double>> x, Iterable<double> w, Iterable<double> y,
    Iterable<double> gradient}) {
  when(costFunctionMock.getGradient(
      x == null ? any : argThat(equals(x)),
      w == null ? any : argThat(equals(w)),
      y == null ? any : argThat(equals(y)),
  )).thenReturn(Float32x4VectorFactory.from(gradient ?? []));
}

void verifyNeverGetGradientCall({Iterable<Iterable<double>> x, Iterable<double> w, Iterable<double> y}) {
  verifyNever(costFunctionMock.getGradient(
    argThat(equals(x)),
    argThat(equals(w)),
    argThat(equals(y)),
  ));
}

void verifyGetGradientCall({Iterable<Iterable<double>> x, Iterable<double> w, Iterable<double> y,
  int callCount}) {
  verify(costFunctionMock.getGradient(
      argThat(equals(x)),
      argThat(equals(w)),
      argThat(equals(y)),
  )).called(callCount);
}

void main() {
  group('Gradient descent optimizer', () {
    setUp(() {
      randomizerMock = RandomizerMock();
      learningRateGeneratorMock = createLearningRateGenerator();
      initialWeightsGeneratorMock = createInitialWeightsGenerator();
      costFunctionMock = CostFunctionMock();
    });

    test('should properly process `batchSize` parameter when the latter is equal to `1` (stochastic case)', () {
      final points = getPoints();
      final labels = Float32x4VectorFactory.from([10.0, 20.0, 30.0, 40.0]);
      final optimizer = createOptimizer(minCoeffUpdate: 1e-100, iterationsLimit: 3, batchSize: 1);

      when(randomizerMock.getIntegerInterval(0, 4, intervalLength: 1)).thenReturn([2, 3]);

      mockGetGradient(x: [[10.0, 20.0, 30.0]], w: [0.0, 0.0, 0.0], y: [30.0], gradient: [10.0, 10.0, 10.0]);
      mockGetGradient(x: [[10.0, 20.0, 30.0]], w: [-20.0, -20.0, -20.0], y: [30.0], gradient: [10.0, 10.0, 10.0]);
      mockGetGradient(x: [[10.0, 20.0, 30.0]], w: [-40.0, -40.0, -40.0], y: [30.0], gradient: [10.0, 10.0, 10.0]);

      optimizer.findExtrema(points, labels);

      verifyGetGradientCall(x: [[10.0, 20.0, 30.0]], w: [0.0, 0.0, 0.0], y: [30.0], callCount: 1);
      verifyGetGradientCall(x: [[10.0, 20.0, 30.0]], w: [-20.0, -20.0, -20.0], y: [30.0], callCount: 1);
      verifyGetGradientCall(x: [[10.0, 20.0, 30.0]], w: [-40.0, -40.0, -40.0], y: [30.0], callCount: 1);
    });

    test('should properly process `batchSize` parameter when the latter is equal to `2` (mini batch case)', () {
      final points = getPoints();
      final labels = Float32x4VectorFactory.from([10.0, 20.0, 30.0, 40.0]);
      final optimizer = createOptimizer(minCoeffUpdate: 1e-100, iterationsLimit: 3, batchSize: 2);

      when(randomizerMock.getIntegerInterval(0, 4, intervalLength: 2)).thenReturn([0, 2]);

      when(costFunctionMock.getGradient(
        argThat(equals([[5.0, 10.0, 15.0], [1.0, 2.0, 3.0]])), any, argThat(equals([10.0, 20.0])))
      ).thenReturn(Float32x4VectorFactory.from([10.0, 10.0, 10.0]));

      optimizer.findExtrema(points, labels);

      verify(costFunctionMock.getGradient(argThat(equals([
        points.getRowVector(0),
        points.getRowVector(1),
      ])), any, argThat(equals([10.0, 20.0]))))
      .called(3); // 3 iterations
    });

    test('should properly process `batchSize` parameter when the latter is equal to `4` (batch case)', () {
      final points = getPoints();
      final labels = Float32x4VectorFactory.from([10.0, 20.0, 30.0, 40.0]);
      final optimizer = createOptimizer(minCoeffUpdate: 1e-100, iterationsLimit: 3, batchSize: 4);

      when(randomizerMock.getIntegerInterval(0, 4, intervalLength: 4)).thenReturn([0, 4]);
      when(costFunctionMock.getGradient(argThat(equals(points)), any, argThat(equals(labels))))
          .thenReturn(Float32x4VectorFactory.from([10.0, 10.0, 10.0]));

      optimizer.findExtrema(points, labels);

      verify(costFunctionMock.getGradient(argThat(equals(points)), any, argThat(equals(labels))))
          .called(3); // 3 iterations
    });

    test('should cut the batch size parameter to make it equal to the number of given points', () {
      final iterationLimit = 3;
      final points = getPoints();
      final labels = Float32x4VectorFactory.from([10.0, 20.0, 30.0, 40.0]);
      final optimizer = createOptimizer(minCoeffUpdate: 1e-100, iterationsLimit: iterationLimit, batchSize: 15);

      when(randomizerMock.getIntegerInterval(0, 4, intervalLength: 4)).thenReturn([0, 4]);
      when(costFunctionMock.getGradient(any, any, any)).thenReturn(Float32x4VectorFactory.from([10.0, 10.0, 10.0]));

      optimizer.findExtrema(points, labels);

      verify(costFunctionMock.getGradient(argThat(equals(points)), any, argThat(equals(labels)))).called(iterationLimit);
      verify(learningRateGeneratorMock.getNextValue()).called(iterationLimit);
    });

    test('should find optimal coefficient values', () {
      final points = Float32x4MatrixFactory.from([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
      ]);
      final labels = Float32x4VectorFactory.from([7.0, 8.0]);
      final optimizer = createOptimizer(minCoeffUpdate: 1e-100, iterationsLimit: 2, batchSize: 2, lambda: 0.0);

      when(randomizerMock.getIntegerInterval(0, 2, intervalLength: 2)).thenReturn([0, 2]);
      when(costFunctionMock.getGradient(
          argThat(equals(points)),
          any,
          argThat(equals(labels))
      )).thenReturn(Float32x4VectorFactory.from([8.0, 8.0, 8.0]));

      final optimalCoefficients = optimizer.findExtrema(points, labels);

      // iteration 1:
      // gradient = [8, 8, 8]
      //
      // c_1 = c_1_prev - eta * partial = 0 - 2 * 8 = -16
      // c_2 = c_2_prev - eta * partial = 0 - 2 * 8 = -16
      // c_3 = c_3_prev - eta * partial = 0 - 2 * 8 = -16
      //
      // c = [-16, -16, -16]
      //
      // iteration 2:
      // gradient = [8, 8, 8]
      //
      // c_1 = c_1_prev - eta * partial_1 = -16 - 2 * 8 = -32
      // c_2 = c_2_prev - eta * partial_2 = -16 - 2 * 8 = -32
      // c_3 = c_3_prev - eta * partial_3 = -16 - 2 * 8 = -32
      //
      // c = [-32, -32, -32]
      //
      expect(optimalCoefficients.toList(), equals([-32.0, -32.0, -32.0]));
    });

    test('should find optimal coefficient values and regularize it', () {
      final points = Float32x4MatrixFactory.from([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
      ]);
      final labels = Float32x4VectorFactory.from([7.0, 8.0]);
      final optimizer = createOptimizer(minCoeffUpdate: 1e-100, iterationsLimit: 3, batchSize: 2, lambda: 10.0);

      when(randomizerMock.getIntegerInterval(0, 2, intervalLength: 2)).thenReturn([0, 2]);
      when(costFunctionMock.getGradient(
          argThat(equals(points)),
          any,
          argThat(equals(labels))
      )).thenReturn(Float32x4VectorFactory.from([8.0, 8.0, 8.0]));

      final optimalCoefficients = optimizer.findExtrema(points, labels);

      // iteration 1:
      // gradient = [8, 8, 8]
      //
      // c_1 = (1 - 2 * eta * lambda) * c_1_prev - eta * partial = 0 - 2 * 8 = -16
      // c_2 = (1 - 2 * eta * lambda) * c_2_prev - eta * partial = 0 - 2 * 8 = -16
      // c_3 = (1 - 2 * eta * lambda) * c_3_prev - eta * partial = 0 - 2 * 8 = -16
      //
      // c = [-16, -16, -16]
      //
      // iteration 2:
      // gradient = [8, 8, 8]
      //
      // c_1 = (1 - 2 * eta * lambda) * c_1_prev - eta * partial_1 = (1 - 2 * 2 * 10) * -16 - 2 * 8 = -39 * -16 - 16 = 608
      // c_2 = (1 - 2 * eta * lambda) * c_2_prev - eta * partial_2 = (1 - 2 * 2 * 10) * -16 - 2 * 8 = -39 * -16 - 16 = 608
      // c_3 = (1 - 2 * eta * lambda) * c_3_prev - eta * partial_3 = (1 - 2 * 2 * 10) * -16 - 2 * 8 = -39 * -16 - 16 = 608
      //
      // c = [608.0, 608.0, 608.0]
      //
      // iteration 3:
      // gradient_1 = [5, 5, 5]
      // gradient_2 = [3, 3, 3]
      // gradient = [8, 8, 8]
      //
      // c_1 = (1 - 2 * eta * lambda) * c_1_prev - eta * partial_1 = (1 - 2 * 2 * 10) * 608 - 2 * 8 = -39 * 608 - 16 = -23728
      // c_2 = (1 - 2 * eta * lambda) * c_2_prev - eta * partial_2 = (1 - 2 * 2 * 10) * 608 - 2 * 8 = -39 * 608 - 16 = -23728
      // c_3 = (1 - 2 * eta * lambda) * c_3_prev - eta * partial_3 = (1 - 2 * 2 * 10) * 608 - 2 * 8 = -39 * 608 - 16 = -23728
      //
      // c = [-23728.0, -23728.0, -23728.0]
      //
      expect(optimalCoefficients.toList(), equals([-23728.0, -23728.0, -23728.0]));
    });

    test('should consider `iterationLimit` parameter', () {
      const maxIteration = 2000;

      final points = Float32x4MatrixFactory.from([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
      ]);
      final labels = Float32x4VectorFactory.from([7.0, 8.0]);
      final optimizer =
          createOptimizer(minCoeffUpdate: 1e-100, iterationsLimit: maxIteration, batchSize: 2, lambda: 0.0);
      when(randomizerMock.getIntegerInterval(0, 2, intervalLength: 2)).thenReturn([0, 2]);
      when(costFunctionMock.getGradient(
          argThat(equals(points)),
          any,
          argThat(equals(labels)),
      )).thenReturn(Float32x4VectorFactory.from([8.0, 8.0, 8.0]));

      optimizer.findExtrema(points, labels);

      verify(learningRateGeneratorMock.getNextValue()).called(maxIteration);
    });

    test('should consider `minCoefficientsUpdate` parameter', () {
      final minCoefficientsUpdate = 4.0;
      final points = Float32x4MatrixFactory.from([
        [1.0, 2.0, 3.0],
      ]);
      final labels = Float32x4VectorFactory.from([1.0]);
      final optimizer =
          createOptimizer(minCoeffUpdate: minCoefficientsUpdate, iterationsLimit: 1000000, batchSize: 1, lambda: 0.0);

      when(randomizerMock.getIntegerInterval(0, 1, intervalLength: 1)).thenReturn([0, 1]);
      when(costFunctionMock.getGradient(
          argThat(equals(points)),
          argThat(equals([0.0, 0.0, 0.0])),
          argThat(equals(labels)),
      )).thenReturn(Float32x4VectorFactory.from([5.0, 5.0, 5.0]));

      when(costFunctionMock.getGradient(
        argThat(equals(points)),
        argThat(equals([-10.0, -10.0, -10.0])),
        argThat(equals(labels)),
      )).thenReturn(Float32x4VectorFactory.from([2.0, 2.0, 2.0]));

      when(costFunctionMock.getGradient(
        argThat(equals(points)),
        argThat(equals([-14.0, -14.0, -14.0])),
        argThat(equals(labels)),
      )).thenReturn(Float32x4VectorFactory.from([1.0, 1.0, 1.0]));

      // c_1 = c_1_prev - eta * partial = 0 - 2 * 5 = -10
      // c_2 = c_2_prev - eta * partial = 0 - 2 * 5 = -10
      // c_3 = c_3_prev - eta * partial = 0 - 2 * 5 = -10
      //
      // c = [-10, -10, -10]
      // distance = sqrt((0 - (-10))^2 + (0 - (-10))^2 + (0 - (-10))^2) = sqrt(300) ~~ 17.32
      //
      // c_1 = c_1_prev - eta * partial = -10 - 2 * 2 = -14
      // c_2 = c_2_prev - eta * partial = -10 - 2 * 2 = -14
      // c_3 = c_3_prev - eta * partial = -10 - 2 * 2 = -14
      //
      // c = [-14, -14, -14]
      // distance = sqrt((-10 - (-14))^2 + (-10 - (-14))^2 + (-10 - (-14))^2) = sqrt(48) ~~ 6.92
      //
      // c_1 = c_1_prev - eta * partial = -14 - 2 * 1 = -16
      // c_2 = c_2_prev - eta * partial = -14 - 2 * 1 = -16
      // c_3 = c_3_prev - eta * partial = -14 - 2 * 1 = -16
      //
      // c = [-16, -16, -16]
      // distance = sqrt((-14 - (-16))^2 + (-14 - (-16))^2 + (-14 - (-16))^2) = sqrt(12) ~~ 3.46
      final coefficients = optimizer.findExtrema(points, labels);
      verify(learningRateGeneratorMock.getNextValue()).called(3);
      expect(coefficients.toList(), equals([-16.0, -16.0, -16.0]));
    });

    test('should consider `learningRate` parameter', () {
      final initialLearningRate = 10.0;
      createOptimizer(minCoeffUpdate: 1e-100, iterationsLimit: 2, batchSize: 1, lambda: 0.0, eta: initialLearningRate);
      verify(learningRateGeneratorMock.init(initialLearningRate)).called(1);
    });
  });
}
