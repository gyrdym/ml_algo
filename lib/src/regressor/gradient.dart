import 'package:dart_ml/src/cost_function/cost_function_factory.dart';
import 'package:dart_ml/src/math/randomizer/randomizer_factory.dart';
import 'package:dart_ml/src/optimizer/gradient.dart';
import 'package:dart_ml/src/optimizer/initial_weights_generator/initial_weights_generator_factory.dart';
import 'package:dart_ml/src/optimizer/learning_rate_generator/generator_factory.dart';
import 'package:dart_ml/src/optimizer/learning_rate_generator/type.dart';
import 'package:dart_ml/src/regressor/gradient_type.dart';
import 'package:dart_ml/src/regressor/linear_regressor.dart';

class GradientRegressor extends LinearRegressor {
  GradientRegressor({
    int iterationLimit,
    LearningRateType learningRateType = LearningRateType.decreasing,
    double learningRate,
    double minWeightsUpdate,
    double lambda,
    GradientType type = GradientType.miniBatch,
    bool fitIntercept = false,
    double interceptScale = 1.0,
    int randomSeed,
    int batchSize = 1
  }) : super(
      GradientOptimizer(
        RandomizerFactory.defaultRandomizer(randomSeed),
        CostFunctionFactory.squared(),
        LearningRateGeneratorFactory.createByType(learningRateType),
        InitialWeightsGeneratorFactory.zeroWeights(),

        initialLearningRate: learningRate,
        minCoefficientsUpdate: minWeightsUpdate,
        iterationLimit: iterationLimit,
        lambda: lambda,
        batchSize: type == GradientType.stochastic
          ? 1
          : type == GradientType.miniBatch
            ? batchSize
            : double.infinity.toInt()
      ),
      fitIntercept ? interceptScale : 0.0
    );
}
