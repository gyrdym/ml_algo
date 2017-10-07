import 'dart:typed_data';
import 'package:di/di.dart';
import 'package:test/test.dart';
import 'package:mockito/mockito.dart';
import 'package:simd_vector/vector.dart';
import 'package:dart_ml/src/interface.dart';
import 'package:dart_ml/src/implementation.dart';

const int ITERATIONS_NUMBER = 3;

class SGDRandomizerMock extends Mock implements Randomizer {}
class InitialWeightsGeneratorMock extends Mock implements InitialWeightsGenerator {}
class LearningRateGeneratorMock extends Mock implements LearningRateGenerator {}
class GradientCalculatorMock extends Mock implements GradientCalculator {}

void main() {
  group('Mini batch gradient descent optimizer', () {
    LearningRateGenerator learningRateGeneratorMock;
    Randomizer randomizerMock;
    GradientCalculator gradientCalculator;

    SGDOptimizer optimizer;
    List<Float32x4Vector> data;
    Float32List target;

    setUp(() {
      randomizerMock = new SGDRandomizerMock();
      learningRateGeneratorMock = new LearningRateGeneratorMock();
      gradientCalculator = new GradientCalculatorMock();

      injector = new ModuleInjector([
        new Module()
          ..bind(Randomizer, toValue: randomizerMock)
          ..bind(InitialWeightsGenerator, toFactory: () => new InitialWeightsGeneratorMock())
          ..bind(LearningRateGenerator, toValue: learningRateGeneratorMock)
          ..bind(GradientCalculator, toValue: gradientCalculator)
      ]);

      optimizer = GradientOptimizerFactory.createStochasticOptimizer(1e-5, null, ITERATIONS_NUMBER, null, .00001, 0.0001);

      data = [
        new Float32x4Vector.from([230.1, 37.8, 69.2]),
        new Float32x4Vector.from([44.5, 39.3, 45.7]),
        new Float32x4Vector.from([54.5, 29.3, 25.1]),
        new Float32x4Vector.from([41.7, 34.1, 55.5])
      ];

      target = new Float32List.fromList([22.1, 10.4, 20.0, 30.0]);

      when(learningRateGeneratorMock.getNextValue()).thenReturn(1.0);
      when(gradientCalculator.getGradient(argThat(isNotNull), data[0], target[0]))
          .thenReturn(new Float32x4Vector.from([1.0, 1.0, 1.0]));
      when(gradientCalculator.getGradient(argThat(isNotNull), data[1], target[1]))
          .thenReturn(new Float32x4Vector.from([0.0, 0.0, 0.0]));
      when(gradientCalculator.getGradient(argThat(isNotNull), data[2], target[2]))
          .thenReturn(new Float32x4Vector.from([0.01, 0.01, 0.01]));
      when(gradientCalculator.getGradient(argThat(isNotNull), data[3], target[3]))
          .thenReturn(new Float32x4Vector.from([100.0, 100.0, 0.00001]));

      when(randomizerMock.getIntegerFromInterval(0, 4)).thenReturn(0);
    });

    test('should find optimal weights for the given data', () {
      Float32x4Vector weights = optimizer.findMinima(data, target, weights: new Float32x4Vector.from([0.0, 0.0, 0.0]));
      List<double> formattedWeights = weights.asList().map((double value) => double.parse(value.toStringAsFixed(2)))
          .toList();

      verify(randomizerMock.getIntegerFromInterval(0, 4)).called(ITERATIONS_NUMBER);
      verify(learningRateGeneratorMock.getNextValue()).called(ITERATIONS_NUMBER);

      verify(gradientCalculator.getGradient(argThat(isNotNull), data[0], target[0]))
          .called(ITERATIONS_NUMBER);

      verifyNever(gradientCalculator.getGradient(argThat(isNotNull), data[1], target[1]));
      verifyNever(gradientCalculator.getGradient(argThat(isNotNull), data[2], target[2]));
      verifyNever(gradientCalculator.getGradient(argThat(isNotNull), data[3], target[3]));
    });
  });
}
