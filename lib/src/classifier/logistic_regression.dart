import 'package:dart_ml/src/classifier/classifier.dart';
import 'package:dart_ml/src/loss_function/loss_function_factory.dart';
import 'package:dart_ml/src/math/math_analysis/gradient_calculator_factory.dart';
import 'package:dart_ml/src/math/randomizer/randomizer_factory.dart';
import 'package:dart_ml/src/optimizer/gradient.dart';
import 'package:dart_ml/src/optimizer/initial_weights_generator/initial_weights_generator_factory.dart';
import 'package:dart_ml/src/optimizer/learning_rate_generator/learning_rate_generator_factory.dart';

class LogisticRegressor extends Classifier {
  LogisticRegressor({
    int iterationLimit,
    double learningRate,
    double minWeightsDistance,
    double alpha,
    double lambda,
    double argumentIncrement,
    int batchSize = 1
  }) : super(
    new GradientOptimizer(
      RandomizerFactory.Default(),
      LossFunctionFactory.Logistic(),
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
