import 'dart:typed_data';

import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/cost_function/cost_function_factory.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/math/randomizer/randomizer.dart';
import 'package:ml_algo/src/optimizer/gradient/gradient.dart';
import 'package:ml_algo/src/optimizer/gradient/learning_rate_generator/learning_rate_generator.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_generator.dart';
import 'package:ml_linalg/vector.dart';
import 'package:mockito/mockito.dart';

import '../../test_utils/helpers/floating_point_iterable_matchers.dart';
import '../../test_utils/mocks.dart';

Randomizer randomizerMock;
LearningRateGenerator learningRateGeneratorMock;
InitialWeightsGenerator initialWeightsGeneratorMock;
CostFunction costFunctionMock;
CostFunctionFactory costFunctionFactoryMock;

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
  randomizerMock = RandomizerMock();
  learningRateGeneratorMock = createLearningRateGenerator();
  initialWeightsGeneratorMock = createInitialWeightsGenerator();
  costFunctionMock = CostFunctionMock();
  costFunctionFactoryMock = CostFunctionFactoryMock();
  when(costFunctionFactoryMock.fromType(CostFunctionType.squared,
          dtype: Float32x4, scoreToProbMapperType: null))
      .thenReturn(costFunctionMock);

  final randomizerFactoryMock = RandomizerFactoryMock();
  final learningRateGeneratorFactoryMock = LearningRateGeneratorFactoryMock();
  final initialWeightsGeneratorFactory = InitialWeightsGeneratorFactoryMock();

  when(randomizerFactoryMock.create(any)).thenReturn(randomizerMock);
  when(learningRateGeneratorFactoryMock.fromType(any))
      .thenReturn(learningRateGeneratorMock);
  when(initialWeightsGeneratorFactory.fromType(any))
      .thenReturn(initialWeightsGeneratorMock);

  final opt = GradientOptimizer(
      randomizerFactory: randomizerFactoryMock,
      costFunctionFactory: costFunctionFactoryMock,
      costFnType: CostFunctionType.squared,
      learningRateGeneratorFactory: learningRateGeneratorFactoryMock,
      initialWeightsGeneratorFactory: initialWeightsGeneratorFactory,
      initialLearningRate: eta,
      minWeightsUpdate: minCoeffUpdate,
      iterationLimit: iterationsLimit,
      lambda: lambda,
      batchSize: batchSize);

  verify(costFunctionFactoryMock.fromType(CostFunctionType.squared,
      dtype: Float32x4, scoreToProbMapperType: null));

  return opt;
}

void mockGetGradient(CostFunction mock,
    {Iterable<Iterable<double>> x,
    Iterable<double> w,
    Iterable<double> y,
    Iterable<double> gradient}) {
  when(mock.getGradient(
    x == null ? any : argThat(matrixAlmostEqualTo(x)),
    w == null ? any : argThat(vectorAlmostEqualTo(w)),
    y == null ? any : argThat(vectorAlmostEqualTo(y)),
  )).thenReturn(MLVector.from(gradient ?? []));
}
