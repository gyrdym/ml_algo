import 'package:dart_ml/src/cost_function/cost_function_factory.dart';
import 'package:dart_ml/src/math/math_analysis/gradient_calculator_factory.dart';
import 'package:dart_ml/src/math/randomizer/randomizer_factory.dart';
import 'package:dart_ml/src/optimizer/gradient.dart';
import 'package:dart_ml/src/optimizer/initial_weights_generator/initial_weights_generator_factory.dart';
import 'package:dart_ml/src/optimizer/learning_rate_generator/learning_rate_generator_factory.dart';
import 'package:dart_ml/src/regressor/regressor.dart';

class GradientRegressor extends Regressor {
  GradientRegressor({
    int iterationLimit,
    double learningRate,
    double minWeightsDistance,
    double lambda = 0.0,
    double argumentIncrement,
    int batchSize = 1
  }) : super(
      new GradientDescentOptimizer(
        RandomizerFactory.Default(),
        CostFunctionFactory.Squared(),
        GradientCalculatorFactory.Default(),
        LearningRateGeneratorFactory.Simple(),
        InitialWeightsGeneratorFactory.ZeroWeights(),

        learningRate: learningRate,
        minCoefficientsUpdate: minWeightsDistance,
        iterationLimit: iterationLimit,
        lambda: lambda,
        argumentIncrement: argumentIncrement,
        batchSize: batchSize
      )
    );
}
