import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/math/randomizer/randomizer.dart';
import 'package:ml_algo/src/solver/linear/convergence_detector/convergence_detector.dart';
import 'package:ml_algo/src/solver/linear/gradient/gradient.dart';
import 'package:ml_algo/src/solver/linear/gradient/learning_rate_generator/learning_rate_generator.dart';
import 'package:ml_algo/src/solver/linear/initial_weights_generator/initial_weights_generator.dart';
import 'package:ml_algo/src/solver/linear/linear_optimizer.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../../test_utils/helpers/floating_point_iterable_matchers.dart';
import '../../../test_utils/mocks.dart';

Randomizer randomizerMock = RandomizerMock();
LearningRateGenerator learningRateGeneratorMock;
InitialWeightsGenerator initialWeightsGeneratorMock;
ConvergenceDetector convergenceDetectorMock = ConvergenceDetectorMock();

LearningRateGenerator createLearningRateGenerator() {
  final mock = LearningRateGeneratorMock();
  when(mock.getNextValue()).thenReturn(2.0);
  return mock;
}

InitialWeightsGenerator createInitialWeightsGenerator() {
  final mock = InitialWeightsGeneratorMock();
  when(mock.generate(3)).thenReturn(Vector.fromList([0.0, 0.0, 0.0]));
  return mock;
}

GradientOptimizer createOptimizer(Matrix points, Matrix labels,
    CostFunction costFunction, {
  double eta,
  double minCoeffUpdate,
  int iterationsLimit,
  double lambda,
  int batchSize
}) {
  learningRateGeneratorMock = createLearningRateGenerator();
  initialWeightsGeneratorMock = createInitialWeightsGenerator();

  final randomizerFactoryMock = RandomizerFactoryMock();
  final learningRateGeneratorFactoryMock = LearningRateGeneratorFactoryMock();
  final initialWeightsGeneratorFactoryMock =
      InitialWeightsGeneratorFactoryMock();
  final convergenceDetectorFactoryMock = ConvergenceDetectorFactoryMock();

  when(randomizerFactoryMock.create(any)).thenReturn(randomizerMock);
  when(learningRateGeneratorFactoryMock.fromType(any))
      .thenReturn(learningRateGeneratorMock);
  when(initialWeightsGeneratorFactoryMock.fromType(any, any))
      .thenReturn(initialWeightsGeneratorMock);
  when(convergenceDetectorFactoryMock.create(any, any))
      .thenReturn(convergenceDetectorMock);

  return GradientOptimizer(
      points, labels,
      randomizerFactory: randomizerFactoryMock,
      costFunction: costFunction,
      learningRateGeneratorFactory: learningRateGeneratorFactoryMock,
      initialWeightsGeneratorFactory: initialWeightsGeneratorFactoryMock,
      convergenceDetectorFactory: convergenceDetectorFactoryMock,
      initialLearningRate: eta,
      minCoefficientsUpdate: minCoeffUpdate,
      iterationLimit: iterationsLimit,
      lambda: lambda,
      batchSize: batchSize);
}

void mockGetGradient(CostFunction mock, {
  Iterable<Iterable<double>> x,
  Iterable<Iterable<double>> w,
  Iterable<Iterable<double>> y,
  Matrix gradient
}) {
  when(mock.getGradient(
    x == null ? any : argThat(matrixAlmostEqualTo(x)),
    w == null ? any : argThat(matrixAlmostEqualTo(w)),
    y == null ? any : argThat(matrixAlmostEqualTo(y)),
  )).thenReturn(gradient ?? Matrix.fromList([[]]));
}

void testOptimizer(
  Matrix points, Matrix labels, CostFunction costFunction,
  void callback(LinearOptimizer optimizer), {
  int iterations,
  int batchSize = 1,
  double minCoeffUpdate = 1e-100,
  double lambda = 0.0,
  double eta,
  bool verifyConvergenceDetectorCall = true,
}) {
  when(convergenceDetectorMock.isConverged(
          any, argThat(inInclusiveRange(0, iterations - 1))))
      .thenReturn(false);
  when(convergenceDetectorMock.isConverged(any, iterations)).thenReturn(true);

  final optimizer = createOptimizer(
      points,
      labels,
      costFunction,
      minCoeffUpdate: minCoeffUpdate,
      iterationsLimit: iterations,
      lambda: lambda,
      eta: eta,
      batchSize: batchSize);

  callback(optimizer);

  if (verifyConvergenceDetectorCall) {
    verify(convergenceDetectorMock.isConverged(any, any))
        .called(iterations + 1);
  }
}
