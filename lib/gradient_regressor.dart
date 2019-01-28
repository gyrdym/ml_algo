import 'package:ml_algo/src/cost_function/cost_function_factory_impl.dart';
import 'package:ml_algo/src/math/randomizer/randomizer_factory_impl.dart';
import 'package:ml_algo/src/optimizer/gradient.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_generator_factory_impl.dart';
import 'package:ml_algo/src/optimizer/learning_rate_generator/learning_rate_generator_factory_impl.dart';
import 'package:ml_algo/learning_rate_type.dart';
import 'package:ml_algo/gradient_type.dart';
import 'package:ml_algo/src/regressor/float32x4_linear_regressor.dart';

class GradientRegressor extends Float32x4LinearRegressor {
  GradientRegressor(
      {int iterationLimit,
      LearningRateType learningRateType = LearningRateType.decreasing,
      double learningRate,
      double minWeightsUpdate,
      double lambda,
      GradientType type = GradientType.miniBatch,
      bool fitIntercept = false,
      double interceptScale = 1.0,
      int randomSeed,
      int batchSize = 1})
      : super(
            GradientOptimizer(
                RandomizerFactoryImpl.create(randomSeed),
                CostFunctionFactoryImpl.squared(),
                LearningRateGeneratorFactoryImpl.fromType(learningRateType),
                InitialWeightsGeneratorFactoryImpl.zeroes(),
                initialLearningRate: learningRate,
                minCoefficientsUpdate: minWeightsUpdate,
                iterationLimit: iterationLimit,
                lambda: lambda,
                batchSize: type == GradientType.stochastic
                    ? 1
                    : type == GradientType.miniBatch ? batchSize : double.maxFinite.toInt()),
            fitIntercept ? interceptScale : 0.0);
}
