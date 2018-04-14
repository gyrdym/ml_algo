import 'dart:typed_data';

import 'package:dart_ml/src/core/implementation.dart';
import 'package:dart_ml/src/core/interface.dart';
import 'package:dart_ml/src/di/injector.dart' show coreInjector;
import 'package:di/di.dart';
import 'package:mockito/mockito.dart';
import 'package:simd_vector/vector.dart';
import 'package:test/test.dart';

const int ITERATIONS_NUMBER = 3;

class MBGDRandomizerMock extends Mock implements Randomizer {}
class InitialWeightsGeneratorMock extends Mock implements InitialWeightsGenerator {}
class LearningRateGeneratorMock extends Mock implements LearningRateGenerator {}
class GradientCalculatorMock extends Mock implements GradientCalculator {}
class LossFunctionMock extends Mock implements LossFunction {}
class ScoreFunctionMock extends Mock implements ScoreFunction {}

void main() {
  group('Mini batch gradient descent optimizer', () {
    const lambda = .000001;
    const delta = .00001;

    final point1 = new Float32x4Vector.from([230.1, 37.8, 69.2]);
    final point2 = new Float32x4Vector.from([44.5, 39.3, 45.7]);
    final point3 = new Float32x4Vector.from([54.5, 29.3, 25.1]);
    final point4 = new Float32x4Vector.from([41.7, 34.1, 55.5]);

    LearningRateGenerator learningRateGeneratorMock;
    Randomizer randomizerMock;
    GradientCalculator gradientCalculator;
    LossFunctionMock lossFunctionMock;
    ScoreFunctionMock scoreFunctionMock;

    Optimizer optimizer;
    List<Float32x4Vector> data;
    Float32List labels;

    setUp(() {
      randomizerMock = new MBGDRandomizerMock();
      learningRateGeneratorMock = new LearningRateGeneratorMock();
      gradientCalculator = new GradientCalculatorMock();

      coreInjector = new ModuleInjector([
        new Module()
          ..bind(Randomizer, toValue: randomizerMock)
          ..bind(InitialWeightsGenerator, toFactory: () => new InitialWeightsGeneratorMock())
          ..bind(LearningRateGenerator, toValue: learningRateGeneratorMock)
          ..bind(GradientCalculator, toValue: gradientCalculator)
          ..bind(LossFunction, toValue: lossFunctionMock)
          ..bind(ScoreFunction, toValue: scoreFunctionMock)
      ]);

      optimizer = GradientOptimizerFactory.createMiniBatchOptimizer(1e-5, null, ITERATIONS_NUMBER, null, lambda, delta);

      data = [point1, point2, point3, point4];
      labels = new Float32List.fromList([22.1, 10.4, 20.0, 30.0]);

      when(learningRateGeneratorMock.getNextValue()).thenReturn(1.0);
      when(gradientCalculator.getGradient(any, any, [point1], [labels[0], lambda], delta))
          .thenReturn(new Float32x4Vector.from([1.0, 1.0, 1.0]));
      when(gradientCalculator.getGradient(any, any, [point2], [labels[1], lambda], delta))
          .thenReturn(new Float32x4Vector.from([0.0, 0.0, 0.0]));
      when(gradientCalculator.getGradient(any, any, [point3], [labels[2], lambda], delta))
          .thenReturn(new Float32x4Vector.from([0.01, 0.01, 0.01]));
      when(gradientCalculator.getGradient(any, any, [point4], [labels[3], lambda], delta))
          .thenReturn(new Float32x4Vector.from([100.0, 100.0, 0.00001]));
    });

    test('should find optimal weights for the given data', () {
      when(randomizerMock.getIntegerInterval(0, 4)).thenReturn([0, 4]);

      optimizer.findExtrema(data, labels, initialWeights: new Float32x4Vector.from([0.0, 0.0, 0.0]));

      verify(randomizerMock.getIntegerInterval(0, 4)).called(ITERATIONS_NUMBER);
      verify(learningRateGeneratorMock.getNextValue()).called(ITERATIONS_NUMBER);

      verify(gradientCalculator.getGradient(any, any, [point1], [labels[0], lambda], delta))
          .called(ITERATIONS_NUMBER);
      verify(gradientCalculator.getGradient(any, any, [point2], [labels[1], lambda], delta))
          .called(ITERATIONS_NUMBER);
      verify(gradientCalculator.getGradient(any, any, [point3], [labels[2], lambda], delta))
          .called(ITERATIONS_NUMBER);
      verify(gradientCalculator.getGradient(any, any, [point4], [labels[3], lambda], delta))
          .called(ITERATIONS_NUMBER);
    });

    test('should cut off a piece of certain size from the given data', () {
      when(randomizerMock.getIntegerInterval(0, 4)).thenReturn([1, 3]);

      optimizer.findExtrema(data, labels, initialWeights: new Float32x4Vector.from([0.0, 0.0, 0.0]));

      verifyNever(gradientCalculator.getGradient(any, any, [point1], [labels[0], lambda], delta));
      verifyNever(gradientCalculator.getGradient(any, any, [point4], [labels[3], lambda], delta));

      verify(gradientCalculator.getGradient(any, any, [point2], [labels[1], lambda], delta)).called(ITERATIONS_NUMBER);
      verify(gradientCalculator.getGradient(any, any, [point3], [labels[2], lambda], delta)).called(ITERATIONS_NUMBER);
    });

    test('should throw range error if a random range is bigger than data length', () {
      when(randomizerMock.getIntegerInterval(0, 4)).thenReturn([0, 5]);

      expect(() {
        optimizer.findExtrema(data, labels, initialWeights: new Float32x4Vector.from([0.0, 0.0, 0.0]));
      }, throwsRangeError);
    });
  });
}
