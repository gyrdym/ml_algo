import 'dart:typed_data';

import 'package:ml_algo/src/cost_function/cost_function_factory.dart';
import 'package:ml_algo/src/cost_function/cost_function_factory_impl.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/math/randomizer/randomizer_factory.dart';
import 'package:ml_algo/src/math/randomizer/randomizer_factory_impl.dart';
import 'package:ml_algo/src/optimizer/coordinate/coordinate.dart';
import 'package:ml_algo/src/optimizer/gradient/gradient.dart';
import 'package:ml_algo/src/optimizer/gradient/learning_rate_generator/learning_rate_generator_factory.dart';
import 'package:ml_algo/src/optimizer/gradient/learning_rate_generator/learning_rate_generator_factory_impl.dart';
import 'package:ml_algo/src/optimizer/gradient/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_generator_factory.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_generator_factory_impl.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/optimizer/optimizer.dart';
import 'package:ml_algo/src/optimizer/optimizer_factory.dart';
import 'package:ml_algo/src/optimizer/optimizer_type.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper_type.dart';
import 'package:ml_linalg/matrix.dart';

class OptimizerFactoryImpl implements OptimizerFactory {
  const OptimizerFactoryImpl();

  @override
  Optimizer fromType(
    OptimizerType type, Matrix points, Matrix labels, {
    Type dtype,
    RandomizerFactory randomizerFactory = const RandomizerFactoryImpl(),
    CostFunctionFactory costFunctionFactory = const CostFunctionFactoryImpl(),
    LearningRateGeneratorFactory learningRateGeneratorFactory =
        const LearningRateGeneratorFactoryImpl(),
    InitialWeightsGeneratorFactory initialWeightsGeneratorFactory =
        const InitialWeightsGeneratorFactoryImpl(),
    CostFunctionType costFunctionType,
    LearningRateType learningRateType,
    InitialWeightsType initialWeightsType,
    ScoreToProbMapperType scoreToProbMapperType,
    double initialLearningRate,
    double minCoefficientsUpdate,
    int iterationLimit,
    double lambda,
    int batchSize,
    int randomSeed,
  }) {
    switch (type) {
      case OptimizerType.coordinateDescent:
        return coordinate(
          points, labels,
          dtype: dtype,
          initialWeightsGeneratorFactory: initialWeightsGeneratorFactory,
          costFunctionFactory: costFunctionFactory,
          minCoefficientsDiff: minCoefficientsUpdate,
          iterationLimit: iterationLimit,
          lambda: lambda,
          initialWeightsType: initialWeightsType,
          costFunctionType: costFunctionType,
        );

      case OptimizerType.gradientDescent:
        return gradient(
          points, labels,
          dtype: dtype,
          randomizerFactory: randomizerFactory,
          costFunctionFactory: costFunctionFactory,
          learningRateGeneratorFactory: learningRateGeneratorFactory,
          initialWeightsGeneratorFactory: initialWeightsGeneratorFactory,
          costFnType: costFunctionType,
          learningRateType: learningRateType,
          initialWeightsType: initialWeightsType,
          scoreToProbMapperType: scoreToProbMapperType,
          initialLearningRate: initialLearningRate,
          minCoefficientsUpdate: minCoefficientsUpdate,
          iterationLimit: iterationLimit,
          lambda: lambda,
          batchSize: batchSize,
          randomSeed: randomSeed,
        );

      default:
        throw UnimplementedError('Unimplemented optimizer type - $type');
    }
  }

  @override
  Optimizer coordinate(Matrix points, Matrix labels, {
    Type dtype = Float32x4,
    InitialWeightsGeneratorFactory initialWeightsGeneratorFactory =
        const InitialWeightsGeneratorFactoryImpl(),
    CostFunctionFactory costFunctionFactory = const CostFunctionFactoryImpl(),
    double minCoefficientsDiff,
    int iterationLimit,
    double lambda,
    InitialWeightsType initialWeightsType,
    CostFunctionType costFunctionType,
  }) =>
      CoordinateOptimizer(
        points, labels,
        dtype: dtype,
        initialWeightsGeneratorFactory: initialWeightsGeneratorFactory,
        costFunctionFactory: costFunctionFactory,
        minCoefficientsDiff: minCoefficientsDiff,
        iterationsLimit: iterationLimit,
        lambda: lambda,
        initialWeightsType: initialWeightsType,
        costFunctionType: costFunctionType,
      );

  @override
  Optimizer gradient(Matrix points, Matrix labels, {
    Type dtype = Float32x4,
    RandomizerFactory randomizerFactory = const RandomizerFactoryImpl(),
    CostFunctionFactory costFunctionFactory = const CostFunctionFactoryImpl(),
    LearningRateGeneratorFactory learningRateGeneratorFactory =
        const LearningRateGeneratorFactoryImpl(),
    InitialWeightsGeneratorFactory initialWeightsGeneratorFactory =
        const InitialWeightsGeneratorFactoryImpl(),
    CostFunctionType costFnType,
    LearningRateType learningRateType,
    InitialWeightsType initialWeightsType,
    ScoreToProbMapperType scoreToProbMapperType,
    double initialLearningRate,
    double minCoefficientsUpdate,
    int iterationLimit,
    double lambda,
    int batchSize,
    int randomSeed,
  }) =>
      GradientOptimizer(
        points, labels,
        randomizerFactory: randomizerFactory,
        costFunctionFactory: costFunctionFactory,
        learningRateGeneratorFactory: learningRateGeneratorFactory,
        initialWeightsGeneratorFactory: initialWeightsGeneratorFactory,
        costFnType: costFnType,
        learningRateType: learningRateType,
        initialWeightsType: initialWeightsType,
        scoreToProbMapperType: scoreToProbMapperType,
        initialLearningRate: initialLearningRate,
        minCoefficientsUpdate: minCoefficientsUpdate,
        iterationLimit: iterationLimit,
        lambda: lambda,
        batchSize: batchSize,
        randomSeed: randomSeed,
      );
}
