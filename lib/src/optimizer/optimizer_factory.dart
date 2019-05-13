import 'package:ml_algo/src/cost_function/cost_function_factory.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/math/randomizer/randomizer_factory.dart';
import 'package:ml_algo/src/optimizer/gradient/learning_rate_generator/learning_rate_generator_factory.dart';
import 'package:ml_algo/src/optimizer/gradient/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_generator_factory.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/optimizer/optimizer.dart';
import 'package:ml_algo/src/optimizer/optimizer_type.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper_type.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';

abstract class OptimizerFactory {
  Optimizer fromType(
    OptimizerType type, Matrix points, Matrix labels, {
    DType dtype,
    RandomizerFactory randomizerFactory,
    CostFunctionFactory costFunctionFactory,
    LearningRateGeneratorFactory learningRateGeneratorFactory,
    InitialWeightsGeneratorFactory initialWeightsGeneratorFactory,
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
  });

  Optimizer gradient(Matrix points, Matrix labels, {
    DType dtype,
    RandomizerFactory randomizerFactory,
    CostFunctionFactory costFunctionFactory,
    LearningRateGeneratorFactory learningRateGeneratorFactory,
    InitialWeightsGeneratorFactory initialWeightsGeneratorFactory,
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
  });

  Optimizer coordinate(Matrix points, Matrix labels, {
    DType dtype,
    InitialWeightsGeneratorFactory initialWeightsGeneratorFactory,
    CostFunctionFactory costFunctionFactory,
    double minCoefficientsDiff,
    int iterationLimit,
    double lambda,
    InitialWeightsType initialWeightsType,
    CostFunctionType costFunctionType,
  });
}
