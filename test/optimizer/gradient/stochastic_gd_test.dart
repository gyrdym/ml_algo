import 'dart:typed_data';

import 'package:dart_ml/src/core/loss_function/loss_function.dart';
import 'package:dart_ml/src/core/math/math_analysis/gradient_calculator.dart';
import 'package:dart_ml/src/core/math/randomizer/randomizer.dart';
import 'package:dart_ml/src/core/optimizer/gradient/factory.dart';
import 'package:dart_ml/src/core/optimizer/gradient/learning_rate_generator/learning_rate_generator.dart';
import 'package:dart_ml/src/core/optimizer/initial_weights_generator/initial_weights_generator.dart';
import 'package:dart_ml/src/core/optimizer/optimizer.dart';
import 'package:dart_ml/src/core/score_function/score_function.dart';
import 'package:dart_ml/src/di/injector.dart' show coreInjector;
import 'package:di/di.dart';
import 'package:mockito/mockito.dart';
import 'package:simd_vector/vector.dart';
import 'package:test/test.dart';

class RandomizerMock extends Mock implements Randomizer {}
class InitialWeightsGeneratorMock extends Mock implements InitialWeightsGenerator {}
class LearningRateGeneratorMock extends Mock implements LearningRateGenerator {}
class GradientCalculatorMock extends Mock implements GradientCalculator {}
class LossFunctionMock extends Mock implements LossFunction {}
class ScoreFunctionMock extends Mock implements ScoreFunction {}

void main() {
  group('Stochastic gradient descent optimizer', () {
    const int iterationsLimit = 3;
    const eta = 1e-5;
    const lambda = .00001;
    const delta = .0001;

    LearningRateGenerator learningRateGeneratorMock;
    Randomizer randomizerMock;
    GradientCalculator gradientCalculator;
    LossFunctionMock lossFunctionMock;
    ScoreFunctionMock scoreFunctionMock;

    Optimizer optimizer;
    List<Float32x4Vector> data;
    Float32List labels;

    setUp(() {
      randomizerMock = new RandomizerMock();
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

      optimizer = gradientOptimizerFactory(eta, null, iterationsLimit, lambda, delta);

      data = [
        new Float32x4Vector.from([230.1, 37.8, 69.2]),
        new Float32x4Vector.from([44.5, 39.3, 45.7]),
        new Float32x4Vector.from([54.5, 29.3, 25.1]),
        new Float32x4Vector.from([41.7, 34.1, 55.5])
      ];

      labels = new Float32List.fromList([22.1, 10.4, 20.0, 30.0]);

      when(learningRateGeneratorMock.getNextValue()).thenReturn(1.0);
      when(gradientCalculator.getGradient(any, any, [data[0]], [labels[0], lambda], 0.0001))
          .thenReturn(new Float32x4Vector.from([1.0, 1.0, 1.0]));
      when(gradientCalculator.getGradient(any, any, [data[1]], [labels[1], lambda], 0.0001))
          .thenReturn(new Float32x4Vector.from([0.0, 0.0, 0.0]));
      when(gradientCalculator.getGradient(any, any, [data[2]], [labels[2], lambda], 0.0001))
          .thenReturn(new Float32x4Vector.from([0.01, 0.01, 0.01]));
      when(gradientCalculator.getGradient(any, any, [data[3]], [labels[3], lambda], 0.0001))
          .thenReturn(new Float32x4Vector.from([100.0, 100.0, 0.00001]));

      when(randomizerMock.getIntegerInterval(0, 4, intervalLength: 1)).thenReturn([0, 1]);
    });

    test('should find optimal weights for the given data', () {
      optimizer.findExtrema(data, labels, initialWeights: new Float32x4Vector.from([0.0, 0.0, 0.0]));

      verify(randomizerMock.getIntegerInterval(0, 4, intervalLength: 1)).called(iterationsLimit);
      verify(learningRateGeneratorMock.getNextValue()).called(iterationsLimit);

      verify(gradientCalculator.getGradient(any, any, [data[0]], [labels[0], lambda], 0.0001))
          .called(iterationsLimit);

      verifyNever(gradientCalculator.getGradient(any, any, [data[1]], [labels[1], lambda], 0.0001));
      verifyNever(gradientCalculator.getGradient(any, any, [data[2]], [labels[2], lambda], 0.0001));
      verifyNever(gradientCalculator.getGradient(any, any, [data[3]], [labels[3], lambda], 0.0001));
    });
  });
}
