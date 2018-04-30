import 'package:dart_ml/src/cost_function/cost_function.dart';
import 'package:dart_ml/src/math/randomizer/randomizer.dart';
import 'package:dart_ml/src/optimizer/gradient.dart';
import 'package:dart_ml/src/optimizer/initial_weights_generator/initial_weights_generator.dart';
import 'package:dart_ml/src/optimizer/learning_rate_generator/learning_rate_generator.dart';
import 'package:mockito/mockito.dart';
import 'package:simd_vector/vector.dart';
import 'package:test/test.dart';

class RandomizerMock extends Mock implements Randomizer {}
class InitialWeightsGeneratorMock extends Mock implements InitialWeightsGenerator {}
class LearningRateGeneratorMock extends Mock implements LearningRateGenerator {}
class CostFunctionMock extends Mock implements CostFunction {}

Randomizer randomizerMock;
LearningRateGenerator learningRateGeneratorMock;
CostFunction costFunctionMock;
InitialWeightsGenerator initialWeightsGeneratorMock;

LearningRateGenerator createLearningRateGenerator() {
  final mock = new LearningRateGeneratorMock();
  when(mock.getNextValue()).thenReturn(2.0);
  return mock;
}

InitialWeightsGenerator createInitialWeightsGenerator() {
  final mock = new InitialWeightsGeneratorMock();
  when(mock.generate(3)).thenReturn(new Float32x4Vector.from([0.0, 0.0, 0.0]));
  return mock;
}

List<Float32x4Vector> getMeaninglessData() {
  return [
    new Float32x4Vector.from([5.0, 10.0, 15.0]),
    new Float32x4Vector.from([1.0, 2.0, 3.0]),
    new Float32x4Vector.from([10.0, 20.0, 30.0]),
    new Float32x4Vector.from([100.0, 200.0, 300.0])
  ];
}

GradientOptimizer createOptimizer({
  double eta,
  double minCoeffUpdate,
  int iterationsLimit,
  double lambda,
  int batchSize
}) {
  return new GradientOptimizer(
    randomizerMock,
    costFunctionMock,
    learningRateGeneratorMock,
    initialWeightsGeneratorMock,

    learningRate: eta,
    minCoefficientsUpdate: minCoeffUpdate,
    iterationLimit: iterationsLimit,
    lambda: lambda,
    batchSize: batchSize
  );
}

void main() {
  group('Gradient descent optimizer', () {
    setUp(() {
      randomizerMock = new RandomizerMock();
      learningRateGeneratorMock = createLearningRateGenerator();
      initialWeightsGeneratorMock = createInitialWeightsGenerator();
      costFunctionMock = new CostFunctionMock();
    });

    test('should properly process `batchSize` parameter when the latter is equal to `1` (stochastic case)', () {
      final points = getMeaninglessData();
      final labels = new Float32x4Vector.from([10.0, 20.0, 30.0, 40.0]);
      final optimizer = createOptimizer(minCoeffUpdate: 1e-100, iterationsLimit: 3, batchSize: 1);

      when(randomizerMock.getIntegerInterval(0, 4, intervalLength: 1)).thenReturn([2, 3]);
      when(costFunctionMock.getPartialDerivative(argThat(inInclusiveRange(0, 3)), points[2], any, labels[2]))
          .thenReturn(10.0);

      optimizer.findExtrema(points, labels);

      verifyNever(costFunctionMock.getPartialDerivative(argThat(inInclusiveRange(0, 3)), points[0], any, labels[0]));
      verifyNever(costFunctionMock.getPartialDerivative(argThat(inInclusiveRange(0, 3)), points[1], any, labels[1]));
      verifyNever(costFunctionMock.getPartialDerivative(argThat(inInclusiveRange(0, 3)), points[3], any, labels[3]));

      verify(costFunctionMock.getPartialDerivative(argThat(inInclusiveRange(0, 3)), points[2], any, labels[2]))
          .called(3 * 3); // 3 iterations 3 times in each iteration
    });

    test('should properly process `batchSize` parameter when the latter is equal to `2` (mini batch case)', () {
      final points = getMeaninglessData();
      final labels = new Float32x4Vector.from([10.0, 20.0, 30.0, 40.0]);
      final optimizer = createOptimizer(minCoeffUpdate: 1e-100, iterationsLimit: 3, batchSize: 2);

      when(randomizerMock.getIntegerInterval(0, 4, intervalLength: 2)).thenReturn([0, 2]);

      when(costFunctionMock.getPartialDerivative(argThat(inInclusiveRange(0, 3)), points[0], any, labels[0]))
          .thenReturn(10.0);
      when(costFunctionMock.getPartialDerivative(argThat(inInclusiveRange(0, 3)), points[1], any, labels[1]))
          .thenReturn(20.0);

      optimizer.findExtrema(points, labels);

      verifyNever(costFunctionMock.getPartialDerivative(argThat(inInclusiveRange(0, 3)), points[2], any, labels[2]));
      verifyNever(costFunctionMock.getPartialDerivative(argThat(inInclusiveRange(0, 3)), points[3], any, labels[3]));

      verify(costFunctionMock.getPartialDerivative(argThat(inInclusiveRange(0, 3)), points[0], any, labels[0]))
          .called(3 * 3); // 3 iterations 3 times in each iteration
      verify(costFunctionMock.getPartialDerivative(argThat(inInclusiveRange(0, 3)), points[1], any, labels[1]))
          .called(3 * 3); // 3 iterations 3 times in each iteration
    });

    test('should properly process `batchSize` parameter when the latter is equal to `4` (batch case)', () {
      final points = getMeaninglessData();
      final labels = new Float32x4Vector.from([10.0, 20.0, 30.0, 40.0]);
      final optimizer = createOptimizer(minCoeffUpdate: 1e-100, iterationsLimit: 3, batchSize: 4);

      when(randomizerMock.getIntegerInterval(0, 4, intervalLength: 4)).thenReturn([0, 4]);

      when(costFunctionMock.getPartialDerivative(argThat(inInclusiveRange(0, 3)), points[0], any, labels[0]))
          .thenReturn(10.0);
      when(costFunctionMock.getPartialDerivative(argThat(inInclusiveRange(0, 3)), points[1], any, labels[1]))
          .thenReturn(20.0);
      when(costFunctionMock.getPartialDerivative(argThat(inInclusiveRange(0, 3)), points[2], any, labels[2]))
          .thenReturn(30.0);
      when(costFunctionMock.getPartialDerivative(argThat(inInclusiveRange(0, 3)), points[3], any, labels[3]))
          .thenReturn(40.0);

      optimizer.findExtrema(points, labels);

      verify(costFunctionMock.getPartialDerivative(argThat(inInclusiveRange(0, 3)), points[0], any, labels[0]))
          .called(3 * 3); // 3 iterations 3 times in each iteration
      verify(costFunctionMock.getPartialDerivative(argThat(inInclusiveRange(0, 3)), points[1], any, labels[1]))
          .called(3 * 3); // 3 iterations 3 times in each iteration
      verify(costFunctionMock.getPartialDerivative(argThat(inInclusiveRange(0, 3)), points[2], any, labels[2]))
          .called(3 * 3); // 3 iterations 3 times in each iteration
      verify(costFunctionMock.getPartialDerivative(argThat(inInclusiveRange(0, 3)), points[3], any, labels[3]))
          .called(3 * 3); // 3 iterations 3 times in each iteration
    });

    test('should find optimal coefficient values', () {
      final points = <Float32x4Vector>[
        new Float32x4Vector.from([1.0, 2.0, 3.0]),
        new Float32x4Vector.from([4.0, 5.0, 6.0])
      ];
      final labels = new Float32x4Vector.from([7.0, 8.0]);
      final optimizer = createOptimizer(minCoeffUpdate: 1e-100, iterationsLimit: 2, batchSize: 2, lambda: 0.0);

      when(randomizerMock.getIntegerInterval(0, 2, intervalLength: 2)).thenReturn([0,2]);

      when(costFunctionMock.getPartialDerivative(argThat(inInclusiveRange(0, 3)), points[0], any, labels[0]))
          .thenReturn(5.0);
      when(costFunctionMock.getPartialDerivative(argThat(inInclusiveRange(0, 3)), points[1], any, labels[1]))
          .thenReturn(3.0);

      final optimalCoefficients = optimizer.findExtrema(points, labels);

      // iteration 1:
      // gradient_1 = [5, 5, 5]
      // gradient_2 = [3, 3, 3]
      // gradient = [8, 8, 8]
      //
      // c_1 = c_1_prev - eta * partial = 0 - 2 * 8 = -16
      // c_2 = c_2_prev - eta * partial = 0 - 2 * 8 = -16
      // c_3 = c_3_prev - eta * partial = 0 - 2 * 8 = -16
      //
      // c = [-16, -16, -16]
      //
      // iteration 2:
      // gradient_1 = [5, 5, 5]
      // gradient_2 = [3, 3, 3]
      // gradient = [8, 8, 8]
      //
      // c_1 = c_1_prev - eta * partial_1 = -16 - 2 * 8 = -32
      // c_2 = c_2_prev - eta * partial_2 = -16 - 2 * 8 = -32
      // c_3 = c_3_prev - eta * partial_3 = -16 - 2 * 8 = -32
      //
      // c = [-32, -32, -32]
      //
      expect(optimalCoefficients, equals([-32.0, -32.0, -32.0]));
    });

    test('should find optimal coefficient values and regularize it', () {
      final points = <Float32x4Vector>[
        new Float32x4Vector.from([1.0, 2.0, 3.0]),
        new Float32x4Vector.from([4.0, 5.0, 6.0])
      ];
      final labels = new Float32x4Vector.from([7.0, 8.0]);
      final optimizer = createOptimizer(minCoeffUpdate: 1e-100, iterationsLimit: 3, batchSize: 2, lambda: 10.0);

      when(randomizerMock.getIntegerInterval(0, 2, intervalLength: 2)).thenReturn([0,2]);

      when(costFunctionMock.getPartialDerivative(argThat(inInclusiveRange(0, 3)), points[0], any, labels[0]))
          .thenReturn(5.0);
      when(costFunctionMock.getPartialDerivative(argThat(inInclusiveRange(0, 3)), points[1], any, labels[1]))
          .thenReturn(3.0);

      final optimalCoefficients = optimizer.findExtrema(points, labels);

      // iteration 1:
      // gradient_1 = [5, 5, 5]
      // gradient_2 = [3, 3, 3]
      // gradient = [8, 8, 8]
      //
      // c_1 = (1 - 2 * eta * lambda) * c_1_prev - eta * partial = 0 - 2 * 8 = -16
      // c_2 = (1 - 2 * eta * lambda) * c_2_prev - eta * partial = 0 - 2 * 8 = -16
      // c_3 = (1 - 2 * eta * lambda) * c_3_prev - eta * partial = 0 - 2 * 8 = -16
      //
      // c = [-16, -16, -16]
      //
      // iteration 2:
      // gradient_1 = [5, 5, 5]
      // gradient_2 = [3, 3, 3]
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
      expect(optimalCoefficients, equals([-23728.0, -23728.0, -23728.0]));
    });

    test('should consider `iterationLimit` parameter', () {
      const maxIteration = 2000;

      final points = <Float32x4Vector>[
        new Float32x4Vector.from([1.0, 2.0, 3.0]),
        new Float32x4Vector.from([4.0, 5.0, 6.0])
      ];
      final labels = new Float32x4Vector.from([7.0, 8.0]);
      final optimizer = createOptimizer(minCoeffUpdate: 1e-100, iterationsLimit: maxIteration, batchSize: 2, lambda: 0.0);
      when(randomizerMock.getIntegerInterval(0, 2, intervalLength: 2)).thenReturn([0,2]);
      when(costFunctionMock.getPartialDerivative(argThat(inInclusiveRange(0, 3)), points[0], any, labels[0]))
          .thenReturn(5.0);
      when(costFunctionMock.getPartialDerivative(argThat(inInclusiveRange(0, 3)), points[1], any, labels[1]))
          .thenReturn(3.0);

      optimizer.findExtrema(points, labels);

      verify(learningRateGeneratorMock.getNextValue()).called(maxIteration);
    });

//    test('should consider `minCoefficientsUpdate` parameter', () {
//      final points = <Float32x4Vector>[new Float32x4Vector.from([1.0, 2.0, 3.0])];
//      final labels = new Float32x4Vector.from([1.0]);
//      final optimizer = createOptimizer(minCoeffUpdate: 1e-100, iterationsLimit: 1e100, batchSize: 2, lambda: 0.0);
//
//      when(randomizerMock.getIntegerInterval(0, 1, intervalLength: 1)).thenReturn([0,1]);
//      when(costFunctionMock.getPartialDerivative(argThat(inInclusiveRange(0, 3)), points[0], any, labels[0]))
//          .thenReturn(5.0);
//      optimizer.findExtrema(points, labels);
//    });
  });
}
