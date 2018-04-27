import 'dart:typed_data';

import 'package:dart_ml/src/cost_function/cost_function.dart';
import 'package:dart_ml/src/math/math_analysis/gradient_calculator.dart';
import 'package:dart_ml/src/math/randomizer/randomizer.dart';
import 'package:dart_ml/src/optimizer/gradient.dart';
import 'package:dart_ml/src/optimizer/learning_rate_generator/learning_rate_generator.dart';
import 'package:dart_ml/src/optimizer/initial_weights_generator/initial_weights_generator.dart';
import 'package:dart_ml/src/optimizer/optimizer.dart';
import 'package:mockito/mockito.dart';
import 'package:simd_vector/vector.dart';
import 'package:test/test.dart';

class RandomizerMock extends Mock implements Randomizer {}
class InitialWeightsGeneratorMock extends Mock implements InitialWeightsGenerator {}
class LearningRateGeneratorMock extends Mock implements LearningRateGenerator {}
class GradientCalculatorMock extends Mock implements GradientCalculator {}
class LossFunctionMock extends Mock implements CostFunction {}

void main() {
  group('Stochastic gradient descent optimizer', () {
    const int iterationsLimit = 3;
    const eta = 1e-5;
    const lambda = .00001;
    const delta = .0001;

    LearningRateGenerator learningRateGeneratorMock;
    Randomizer randomizerMock;
    GradientCalculator gradientCalculatorMock;
    LossFunctionMock lossFunctionMock;
    InitialWeightsGenerator initialWeightsGeneratorMock;

    Optimizer optimizer;
    List<Float32x4Vector> data;
    Float32x4Vector labels;

    setUp(() {
      randomizerMock = new RandomizerMock();
      learningRateGeneratorMock = new LearningRateGeneratorMock();
      gradientCalculatorMock = new GradientCalculatorMock();
      lossFunctionMock = new LossFunctionMock();
      initialWeightsGeneratorMock = new InitialWeightsGeneratorMock();

      optimizer = new GradientOptimizer(
        randomizerMock,
        lossFunctionMock,
        gradientCalculatorMock,
        learningRateGeneratorMock,
        initialWeightsGeneratorMock,

        learningRate: eta,
        minCoefficientsUpdate: null,
        iterationLimit: iterationsLimit,
        lambda: lambda,
        argumentIncrement: delta,
        batchSize: 1
      );

      data = [
        new Float32x4Vector.from([230.1, 37.8, 69.2]),
        new Float32x4Vector.from([44.5, 39.3, 45.7]),
        new Float32x4Vector.from([54.5, 29.3, 25.1]),
        new Float32x4Vector.from([41.7, 34.1, 55.5])
      ];

      labels = new Float32x4Vector.from([22.1, 10.4, 20.0, 30.0]);

      when(learningRateGeneratorMock.getNextValue()).thenReturn(1.0);
      when(gradientCalculatorMock.getGradient(any, any, [data[0]], [labels[0], lambda], 0.0001))
          .thenReturn(new Float32x4Vector.from([1.0, 1.0, 1.0]));
      when(gradientCalculatorMock.getGradient(any, any, [data[1]], [labels[1], lambda], 0.0001))
          .thenReturn(new Float32x4Vector.from([0.0, 0.0, 0.0]));
      when(gradientCalculatorMock.getGradient(any, any, [data[2]], [labels[2], lambda], 0.0001))
          .thenReturn(new Float32x4Vector.from([0.01, 0.01, 0.01]));
      when(gradientCalculatorMock.getGradient(any, any, [data[3]], [labels[3], lambda], 0.0001))
          .thenReturn(new Float32x4Vector.from([100.0, 100.0, 0.00001]));

      when(randomizerMock.getIntegerInterval(0, 4, intervalLength: 1)).thenReturn([0, 1]);
    });

    test('should find optimal weights for the given data', () {
      optimizer.findExtrema(data, labels, initialWeights: new Float32x4Vector.from([0.0, 0.0, 0.0]));

      verify(randomizerMock.getIntegerInterval(0, 4, intervalLength: 1)).called(iterationsLimit);
      verify(learningRateGeneratorMock.getNextValue()).called(iterationsLimit);

      verify(gradientCalculatorMock.getGradient(any, any, [data[0]], [labels[0], lambda], 0.0001))
          .called(iterationsLimit);

      verifyNever(gradientCalculatorMock.getGradient(any, any, [data[1]], [labels[1], lambda], 0.0001));
      verifyNever(gradientCalculatorMock.getGradient(any, any, [data[2]], [labels[2], lambda], 0.0001));
      verifyNever(gradientCalculatorMock.getGradient(any, any, [data[3]], [labels[3], lambda], 0.0001));
    });
  });
}
