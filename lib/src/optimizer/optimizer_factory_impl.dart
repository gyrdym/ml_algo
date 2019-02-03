import 'dart:typed_data';

import 'package:ml_algo/learning_rate_type.dart';
import 'package:ml_algo/src/cost_function/cost_function_factory.dart';
import 'package:ml_algo/src/cost_function/cost_function_factory_impl.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/link_function/link_function_type.dart';
import 'package:ml_algo/src/math/randomizer/randomizer_factory.dart';
import 'package:ml_algo/src/math/randomizer/randomizer_factory_impl.dart';
import 'package:ml_algo/src/optimizer/coordinate.dart';
import 'package:ml_algo/src/optimizer/gradient.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_generator_factory.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_generator_factory_impl.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/optimizer/learning_rate_generator/learning_rate_generator_factory.dart';
import 'package:ml_algo/src/optimizer/learning_rate_generator/learning_rate_generator_factory_impl.dart';
import 'package:ml_algo/src/optimizer/optimizer.dart';
import 'package:ml_algo/src/optimizer/optimizer_factory.dart';

class OptimizerFactoryImpl implements OptimizerFactory {
  const OptimizerFactoryImpl();

  @override
  Optimizer coordinate({
    Type dtype = Float32x4,
    InitialWeightsGeneratorFactory initialWeightsGeneratorFactory = const InitialWeightsGeneratorFactoryImpl(),
    CostFunctionFactory costFunctionFactory = const CostFunctionFactoryImpl(),
    double minCoefficientsDiff,
    int iterationLimit,
    double lambda,
    InitialWeightsType initialWeightsType,
    CostFunctionType costFunctionType,
  }) => CoordinateOptimizer(
    dtype: dtype,
    initialWeightsGeneratorFactory: initialWeightsGeneratorFactory,
    costFunctionFactory: costFunctionFactory,
    minCoefficientsDiff: minCoefficientsDiff,
    iterationLimit: iterationLimit,
    lambda: lambda,
    initialWeightsType: initialWeightsType,
    costFunctionType: costFunctionType,
  );

  @override
  Optimizer gradient({
    RandomizerFactory randomizerFactory = const RandomizerFactoryImpl(),
    CostFunctionFactory costFunctionFactory = const CostFunctionFactoryImpl(),
    LearningRateGeneratorFactory learningRateGeneratorFactory = const LearningRateGeneratorFactoryImpl(),
    InitialWeightsGeneratorFactory initialWeightsGeneratorFactory = const InitialWeightsGeneratorFactoryImpl(),
    CostFunctionType costFnType,
    LearningRateType learningRateType,
    InitialWeightsType initialWeightsType,
    LinkFunctionType linkFunctionType,
    double initialLearningRate,
    double minCoefficientsUpdate,
    int iterationLimit,
    double lambda,
    int batchSize,
    int randomSeed,
  }) => GradientOptimizer(
    randomizerFactory: randomizerFactory,
    costFunctionFactory: costFunctionFactory,
    learningRateGeneratorFactory: learningRateGeneratorFactory,
    initialWeightsGeneratorFactory: initialWeightsGeneratorFactory,
    costFnType: costFnType,
    learningRateType: learningRateType,
    initialWeightsType: initialWeightsType,
    linkFunctionType: linkFunctionType,
    initialLearningRate: initialLearningRate,
    minCoefficientsUpdate: minCoefficientsUpdate,
    iterationLimit: iterationLimit,
    lambda: lambda,
    batchSize: batchSize,
    randomSeed: randomSeed,
  );
}
