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
class LossFunctionMock extends Mock implements CostFunction {}
class GradientCalculatorMock extends Mock implements GradientCalculator {}

//@TODO: reanimate the test when Float32xrVector will implement an iterable

GradientCalculator createGradientCalculator() {
  final mock = new GradientCalculatorMock();

  // first iteration
  when(mock.getGradient(any, /*[0.0, 0.0, 0.0]*/any, /*[[5.0, 10.0, 15.0]]*/any, [10.0], any))
      .thenReturn([1.0, 2.0, 3.0]);
  when(mock.getGradient(any, /*[0.0, 0.0, 0.0]*/any, /*[[1.0, 2.0, 3.0]]*/any, [20.0], any))
      .thenReturn([2.0, 3.0, 4.0]);

  // every new gradient vector is being added to a previous one (by definition of the gradient descent algorithm), so,
  // to get a new update of the weights vector in the iteration it is needed to sum [1.0, 2.0, 3.0] and [2.0, 3.0, 4.0]:
  // 1.0, 2.0, 3.0
  // 2.0, 3.0, 4.0
  // -------------
  // 3.0, 5.0, 7.0 - consider it a final gradient vector
  //
  // weights vector update: [0.0, 0.0, 0.0] - ETA * [3.0, 5.0, 7.0] = [0.0, 0.0, 0.0] - 2 * [3.0, 5.0, 7.0] =
  // = [0.0, 0.0, 0.0] - [6.0, 10.0, 14.0] = [-6.0, -10.0, -14.0] - updated weights

  // second iteration
  when(mock.getGradient(any, /*[-6.0, -10.0, -14.0]*/any, /*[[5.0, 10.0, 15.0]]*/any, [10.0], any))
      .thenReturn([3.0, 4.0, 5.0]);
  when(mock.getGradient(any, /*[-6.0, -10.0, -14.0]*/any, /*[[1.0, 2.0, 3.0]]*/any, [20.0], any))
      .thenReturn([4.0, 5.0, 6.0]);

  // as considered above, sum up all resulting vectors:
  // 3.0, 4.0, 5.0
  // 4.0, 5.0, 6.0
  // -------------
  // 7.0, 9.0, 11.0
  //
  // weights vector update: [-6.0, -10.0, -14.0] - 2 * [7.0, 9.0, 11.0] = [-6.0, -10.0, -14.0] - [14.0, 18.0, 22.0] =
  // = [-20.0, -28.0, -36.0]

  return mock;
}

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
    const delta = .0001; // the value an argument is increased by
    const lambda = .00001; // regularization term
    const batchSize = 2;

    Randomizer randomizerMock;
    LearningRateGenerator learningRateGeneratorMock;
    GradientCalculator gradientCalculatorMock;
    CostFunction lossFunctionMock;
    InitialWeightsGenerator initialWeightsGeneratorMock;

    Optimizer optimizer;
    List<Float32x4Vector> data;
    Float32List labels;

    setUp(() {
      randomizerMock = new RandomizerMock();
      learningRateGeneratorMock = createLearningRateGenerator();
      gradientCalculatorMock = createGradientCalculator();
      initialWeightsGeneratorMock = createInitialWeightsGenerator();
      lossFunctionMock = new LossFunctionMock();

      data = [
        new Float32x4Vector.from([5.0, 10.0, 15.0]),
        new Float32x4Vector.from([1.0, 2.0, 3.0])
      ];
      labels = new Float32List.fromList([10.0, 20.0]);

      optimizer = new GradientDescentOptimizer(
        randomizerMock,
        lossFunctionMock,
        gradientCalculatorMock,
        learningRateGeneratorMock,
        initialWeightsGeneratorMock,

        learningRate: eta,
        minCoefficientsUpdate: null,
        iterationLimit: iterationsNumber,
        lambda: lambda,
        argumentIncrement: delta,
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
