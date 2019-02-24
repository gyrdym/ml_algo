import 'dart:typed_data';

import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/cost_function/cost_function_factory.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/math/randomizer/randomizer.dart';
import 'package:ml_algo/src/optimizer/convergence_detector/convergence_detector.dart';
import 'package:ml_algo/src/optimizer/gradient/gradient.dart';
import 'package:ml_algo/src/optimizer/gradient/learning_rate_generator/learning_rate_generator.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_generator.dart';
import 'package:ml_algo/src/optimizer/optimizer.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../test_utils/helpers/floating_point_iterable_matchers.dart';
import '../../test_utils/mocks.dart';

Randomizer randomizerMock = RandomizerMock();
LearningRateGenerator learningRateGeneratorMock;
InitialWeightsGenerator initialWeightsGeneratorMock;
CostFunction costFunctionMock = CostFunctionMock();
CostFunctionFactory costFunctionFactoryMock;
ConvergenceDetector convergenceDetectorMock = ConvergenceDetectorMock();

LearningRateGenerator createLearningRateGenerator() {
  final mock = LearningRateGeneratorMock();
  when(mock.getNextValue()).thenReturn(2.0);
  return mock;
}

InitialWeightsGenerator createInitialWeightsGenerator() {
  final mock = InitialWeightsGeneratorMock();
  when(mock.generate(3)).thenReturn(MLVector.from([0.0, 0.0, 0.0]));
  return mock;
}

GradientOptimizer createOptimizer(
    {double eta,
    double minCoeffUpdate,
    int iterationsLimit,
    double lambda,
    int batchSize}) {
  learningRateGeneratorMock = createLearningRateGenerator();
  initialWeightsGeneratorMock = createInitialWeightsGenerator();
  costFunctionFactoryMock = CostFunctionFactoryMock();

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
  when(costFunctionFactoryMock.fromType(CostFunctionType.squared,
          dtype: Float32x4, scoreToProbMapperType: null))
      .thenReturn(costFunctionMock);
  when(convergenceDetectorFactoryMock.create(any, any))
      .thenReturn(convergenceDetectorMock);

  final opt = GradientOptimizer(
      randomizerFactory: randomizerFactoryMock,
      costFunctionFactory: costFunctionFactoryMock,
      costFnType: CostFunctionType.squared,
      learningRateGeneratorFactory: learningRateGeneratorFactoryMock,
      initialWeightsGeneratorFactory: initialWeightsGeneratorFactoryMock,
      convergenceDetectorFactory: convergenceDetectorFactoryMock,
      initialLearningRate: eta,
      minCoefficientsUpdate: minCoeffUpdate,
      iterationLimit: iterationsLimit,
      lambda: lambda,
      batchSize: batchSize);

  verify(costFunctionFactoryMock.fromType(CostFunctionType.squared,
      dtype: Float32x4, scoreToProbMapperType: null));

  return opt;
}

void mockGetGradient(CostFunction mock, {
  Iterable<Iterable<double>> x,
  Iterable<Iterable<double>> w,
  Iterable<Iterable<double>> y,
  MLMatrix gradient
}) {
  when(mock.getGradient(
    x == null ? any : argThat(matrixAlmostEqualTo(x)),
    w == null ? any : argThat(matrixAlmostEqualTo(w)),
    y == null ? any : argThat(matrixAlmostEqualTo(y)),
  )).thenReturn(gradient ?? MLMatrix.from([[]]));
}

void testOptimizer(
  Function callback(Optimizer optimizer), {
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
