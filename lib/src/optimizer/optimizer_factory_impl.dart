import 'package:ml_algo/src/cost_function/cost_function.dart';
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
import 'package:ml_algo/src/utils/default_parameter_values.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';

class OptimizerFactoryImpl implements OptimizerFactory {
  const OptimizerFactoryImpl();

  @override
  Optimizer coordinate(Matrix points, Matrix labels, {
    DType dtype = DefaultParameterValues.dtype,
    InitialWeightsGeneratorFactory initialWeightsGeneratorFactory =
        const InitialWeightsGeneratorFactoryImpl(),
    CostFunction costFunction,
    double minCoefficientsDiff,
    int iterationLimit,
    double lambda,
    InitialWeightsType initialWeightsType,
  }) =>
      CoordinateOptimizer(
        points, labels,
        dtype: dtype,
        initialWeightsGeneratorFactory: initialWeightsGeneratorFactory,
        costFunction: costFunction,
        minCoefficientsDiff: minCoefficientsDiff,
        iterationsLimit: iterationLimit,
        lambda: lambda,
        initialWeightsType: initialWeightsType,
      );

  @override
  Optimizer gradient(Matrix points, Matrix labels, {
    DType dtype = DefaultParameterValues.dtype,
    RandomizerFactory randomizerFactory = const RandomizerFactoryImpl(),
    CostFunction costFunction,
    LearningRateGeneratorFactory learningRateGeneratorFactory =
        const LearningRateGeneratorFactoryImpl(),
    InitialWeightsGeneratorFactory initialWeightsGeneratorFactory =
        const InitialWeightsGeneratorFactoryImpl(),
    LearningRateType learningRateType,
    InitialWeightsType initialWeightsType,
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
        costFunction: costFunction,
        learningRateGeneratorFactory: learningRateGeneratorFactory,
        initialWeightsGeneratorFactory: initialWeightsGeneratorFactory,
        learningRateType: learningRateType,
        initialWeightsType: initialWeightsType,
        initialLearningRate: initialLearningRate,
        minCoefficientsUpdate: minCoefficientsUpdate,
        iterationLimit: iterationLimit,
        lambda: lambda,
        batchSize: batchSize,
        randomSeed: randomSeed,
      );
}
