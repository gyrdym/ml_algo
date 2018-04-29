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
      final optimizer = createOptimizer(minCoeffUpdate: 1e10, iterationsLimit: 3, batchSize: 1);

      when(randomizerMock.getIntegerInterval(0, 4, intervalLength: 1)).thenReturn([2, 3]);
      when(costFunctionMock.getPartialDerivative(argThat(inInclusiveRange(0, 3)), points[2], any, labels[2]))
          .thenReturn(10.0);

      optimizer.findExtrema(points, labels);

      verify(costFunctionMock.getPartialDerivative(argThat(inInclusiveRange(0, 3)), points[2], any, labels[2]))
          .called(3);
      verifyNever(costFunctionMock.getPartialDerivative(argThat(inInclusiveRange(0, 3)), points[0], any, labels[0]));
      verifyNever(costFunctionMock.getPartialDerivative(argThat(inInclusiveRange(0, 3)), points[1], any, labels[1]));
      verifyNever(costFunctionMock.getPartialDerivative(argThat(inInclusiveRange(0, 3)), points[3], any, labels[3]));
    });

    test('should properly process `batchSize` parameter when the latter is equal to `2` (mini batch case)', () {
      final points = getMeaninglessData();
      final labels = new Float32x4Vector.from([10.0, 20.0, 30.0, 40.0]);
      final optimizer = createOptimizer(minCoeffUpdate: 1e10, iterationsLimit: 3, batchSize: 2);

      when(randomizerMock.getIntegerInterval(0, 4, intervalLength: 2)).thenReturn([0, 2]);

      when(costFunctionMock.getPartialDerivative(argThat(inInclusiveRange(0, 3)), points[0], any, labels[0]))
          .thenReturn(10.0);
      when(costFunctionMock.getPartialDerivative(argThat(inInclusiveRange(0, 3)), points[1], any, labels[1]))
          .thenReturn(20.0);

      optimizer.findExtrema(points, labels);

      verify(costFunctionMock.getPartialDerivative(argThat(inInclusiveRange(0, 3)), points[0], any, labels[0]))
          .called(3);
      verify(costFunctionMock.getPartialDerivative(argThat(inInclusiveRange(0, 3)), points[1], any, labels[1]))
          .called(3);

      verifyNever(costFunctionMock.getPartialDerivative(argThat(inInclusiveRange(0, 3)), points[2], any, labels[2]));
      verifyNever(costFunctionMock.getPartialDerivative(argThat(inInclusiveRange(0, 3)), points[3], any, labels[3]));
    });
  });
}
