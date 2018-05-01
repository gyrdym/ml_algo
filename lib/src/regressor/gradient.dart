import 'package:dart_ml/src/cost_function/cost_function_factory.dart';
import 'package:dart_ml/src/math/randomizer/randomizer_factory.dart';
import 'package:dart_ml/src/optimizer/gradient.dart';
import 'package:dart_ml/src/optimizer/initial_weights_generator/initial_weights_generator_factory.dart';
import 'package:dart_ml/src/optimizer/learning_rate_generator/learning_rate_generator_factory.dart';
import 'package:dart_ml/src/regressor/gradient_type.dart';
import 'package:dart_ml/src/regressor/regressor.dart';

class GradientRegressor extends Regressor {
  GradientRegressor({
    int iterationLimit,
    double learningRate,
    double minWeightsUpdate,
    double lambda = 0.0,
    GradientType type = GradientType.MiniBatch,
    int batchSize = 1
  }) : super(
      new GradientOptimizer(
        RandomizerFactory.Default(),
        CostFunctionFactory.Squared(),
        LearningRateGeneratorFactory.Simple(),
        InitialWeightsGeneratorFactory.ZeroWeights(),

        initialLearningRate: learningRate,
        minCoefficientsUpdate: minWeightsUpdate,
        iterationLimit: iterationLimit,
        lambda: lambda,
        batchSize: type == GradientType.Stochastic ? 1 : type == GradientType.MiniBatch ? batchSize : double.INFINITY
      )
    );
}
