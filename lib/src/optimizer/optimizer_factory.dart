import 'package:ml_algo/learning_rate_type.dart';
import 'package:ml_algo/src/cost_function/cost_function_factory.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/link_function/link_function_type.dart';
import 'package:ml_algo/src/math/randomizer/randomizer_factory.dart';
import 'package:ml_algo/src/optimizer/gradient/learning_rate_generator/learning_rate_generator_factory.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_generator_factory.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/optimizer/optimizer.dart';
import 'package:ml_algo/src/optimizer/optimizer_type.dart';

abstract class OptimizerFactory {
  Optimizer fromType(OptimizerType type, {
    Type dtype,
    RandomizerFactory randomizerFactory,
    CostFunctionFactory costFunctionFactory,
    LearningRateGeneratorFactory learningRateGeneratorFactory,
    InitialWeightsGeneratorFactory initialWeightsGeneratorFactory,
    CostFunctionType costFunctionType,
    LearningRateType learningRateType,
    InitialWeightsType initialWeightsType,
    LinkFunctionType linkFunctionType,
    double initialLearningRate,
    double minCoefficientsUpdate,
    int iterationLimit,
    double lambda,
    int batchSize,
    int randomSeed,
  });

  Optimizer gradient({
    Type dtype,
    RandomizerFactory randomizerFactory,
    CostFunctionFactory costFunctionFactory,
    LearningRateGeneratorFactory learningRateGeneratorFactory,
    InitialWeightsGeneratorFactory initialWeightsGeneratorFactory,
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
  });

  Optimizer coordinate({
    Type dtype,
    InitialWeightsGeneratorFactory initialWeightsGeneratorFactory,
    CostFunctionFactory costFunctionFactory,
    double minCoefficientsDiff,
    int iterationLimit,
    double lambda,
    InitialWeightsType initialWeightsType,
    CostFunctionType costFunctionType,
  });
}
