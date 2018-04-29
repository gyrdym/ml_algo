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
  group('Mini batch gradient descent optimizer', () {
    const int iterationsLimit = 3;
    const lambda = .000001;
    const eta = 1e-5;
    const batchSize = 2;

    final point1 = new Float32x4Vector.from([230.1, 37.8, 69.2]);
    final point2 = new Float32x4Vector.from([44.5, 39.3, 45.7]);
    final point3 = new Float32x4Vector.from([54.5, 29.3, 25.1]);
    final point4 = new Float32x4Vector.from([41.7, 34.1, 55.5]);

    LearningRateGenerator learningRateGeneratorMock;
    Randomizer randomizerMock;
    CostFunction lossFunctionMock;
    InitialWeightsGenerator initialWeightsGeneratorMock;

    Optimizer optimizer;
    List<Float32x4Vector> data;
    Float32x4Vector labels;

    setUp(() {
      randomizerMock = new RandomizerMock();
      learningRateGeneratorMock = new LearningRateGeneratorMock();
      initialWeightsGeneratorMock = new InitialWeightsGeneratorMock();

      optimizer = new GradientOptimizer(
        randomizerMock,
        lossFunctionMock,
        learningRateGeneratorMock,
        initialWeightsGeneratorMock,

        learningRate: eta,
        minCoefficientsUpdate: null,
        iterationLimit: iterationsLimit,
        lambda: lambda,
        batchSize: batchSize
      );

      data = [point1, point2, point3, point4];
      labels = new Float32x4Vector.from([22.1, 10.4, 20.0, 30.0]);

      when(learningRateGeneratorMock.getNextValue()).thenReturn(1.0);
    });

    test('should find optimal weights for the given data', () {
      when(randomizerMock.getIntegerInterval(0, 4, intervalLength: batchSize)).thenReturn([0, 4]);

      optimizer.findExtrema(data, labels, initialWeights: new Float32x4Vector.from([0.0, 0.0, 0.0]));

      verify(randomizerMock.getIntegerInterval(0, 4, intervalLength: batchSize)).called(iterationsLimit);
      verify(learningRateGeneratorMock.getNextValue()).called(iterationsLimit);
    });

    test('should cut off a piece of certain size from the given data', () {
      when(randomizerMock.getIntegerInterval(0, 4, intervalLength: batchSize)).thenReturn([1, 3]);
      optimizer.findExtrema(data, labels, initialWeights: new Float32x4Vector.from([0.0, 0.0, 0.0]));
    });

    test('should throw range error if a random range is bigger than data length', () {
      when(randomizerMock.getIntegerInterval(0, 4, intervalLength: batchSize)).thenReturn([0, 5]);

      expect(() {
        optimizer.findExtrema(data, labels, initialWeights: new Float32x4Vector.from([0.0, 0.0, 0.0]));
      }, throwsRangeError);
    });
  });
}
