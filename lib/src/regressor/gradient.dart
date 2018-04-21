import 'dart:typed_data';

import 'package:dart_ml/src/loss_function/loss_function_factory.dart';
import 'package:dart_ml/src/math/math_analysis/gradient_calculator_factory.dart';
import 'package:dart_ml/src/math/randomizer/randomizer_factory.dart';
import 'package:dart_ml/src/metric/regression/type.dart';
import 'package:dart_ml/src/optimizer/gradient_descent.dart';
import 'package:dart_ml/src/optimizer/initial_weights_generator/initial_weights_generator_factory.dart';
import 'package:dart_ml/src/optimizer/learning_rate_generator/learning_rate_generator_factory.dart';
import 'package:dart_ml/src/regressor/regressor.dart';
import 'package:simd_vector/vector.dart';

class GradientRegressor extends Regressor {
  GradientRegressor({
    List<Float32x4Vector> features,
    Float32List labels,
    int iterationLimit,
    double learningRate,
    double minWeightsDistance,
    double lambda = 0.0,
    double argumentIncrement,
    RegressionMetricType metric,
    int batchSize = 1
  }) : super(
      new GradientDescentOptimizer(
        RandomizerFactory.Default(),
        LossFunctionFactory.Squared(),
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
