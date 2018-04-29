import 'package:dart_ml/src/cost_function/cost_function.dart';
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
class CostFunctionMock extends Mock implements CostFunction {}

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

void main() {
  group('Batch gradient descent optimizer', () {
    const iterationsNumber = 2;
    const eta = 2.0; // learning rate
    const lambda = .00001; // regularization term
    const batchSize = 2;

    Randomizer randomizerMock;
    LearningRateGenerator learningRateGeneratorMock;
    CostFunction lossFunctionMock;
    InitialWeightsGenerator initialWeightsGeneratorMock;

    Optimizer optimizer;
    List<Float32x4Vector> data;
    Float32x4Vector labels;

    setUp(() {
      randomizerMock = new RandomizerMock();
      learningRateGeneratorMock = createLearningRateGenerator();
      initialWeightsGeneratorMock = createInitialWeightsGenerator();
      lossFunctionMock = new CostFunctionMock();

      data = [
        new Float32x4Vector.from([5.0, 10.0, 15.0]),
        new Float32x4Vector.from([1.0, 2.0, 3.0])
      ];
      labels = new Float32x4Vector.from([10.0, 20.0]);

      optimizer = new GradientOptimizer(
        randomizerMock,
        lossFunctionMock,
        learningRateGeneratorMock,
        initialWeightsGeneratorMock,

        learningRate: eta,
        minCoefficientsUpdate: null,
        iterationLimit: iterationsNumber,
        lambda: lambda,
        batchSize: batchSize
      );
    });

    tearDown(() {
      verify(learningRateGeneratorMock.getNextValue()).called(iterationsNumber);
    });

    /// go through the [data], calculate a gradient for each data point, sum all the gradients column wise,
    /// update the weights vector with the result summed gradient vector times [ETA], repeat this procedure
    /// [ITERATION_NUMBER] times
    ///
    /// First iteration:
    /// 0. define hyper parameters ETA (1e-5), DELTA (.0001)
    /// 1. get first data point - [5.0, 10.0, 15.0] (x)
    /// 2. get current weight vector - [0.0, 0.0, 0.0] (w)
    /// 3. get y corresponding to the data point - 10.0 (y)
    /// 4. form a cost function - (y - x * w)^2
    /// 5. get a derivative of this function with respect to each of w vector component via normalized derivative formula
    ///    (implemented in the [_GradientCalculatorImpl]):
    ///    cost_function(w + DELTA) - cost_function(w - DELTA) / 2 / DELTA
    ///    cost_function([0.0, 0.0, 0.0] + 0.0001) - cost_function([0.0, 0.0, 0.0] - 0.0001) / 2 / 0.0001 (x = [5.0, 10.0, 15.0])
    /// 6. update w with the value from above - [-6.0, -10.0, -14.0]
    ///
    /// Second iteration
    /// 1. ...
    /// ...
    /// 6. update w with the value - [-20.0, -28.0, -36.0]
    test('should find optimal weights for the given data', () {
      Float32x4Vector weights = optimizer.findExtrema(data, labels);
      List<double> formattedWeights = weights.map((double value) => double.parse(value.toStringAsFixed(2)))
          .toList();
      expect(formattedWeights, equals([-20.0, -28.0, -36.0]));
    }, skip: true);
  });
}
