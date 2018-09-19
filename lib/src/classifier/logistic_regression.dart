import 'package:dart_ml/src/classifier/linear_classifier.dart';
import 'package:dart_ml/src/classifier/multinomial_type.dart';
import 'package:dart_ml/src/cost_function/cost_function_factory.dart';
import 'package:dart_ml/src/optimizer/learning_rate_generator/type.dart';
import 'package:dart_ml/src/score_to_prob_link_function/link_function.dart' as scoreToProbabilityLink;
import 'package:dart_ml/src/math/randomizer/randomizer_factory.dart';
import 'package:dart_ml/src/optimizer/gradient.dart';
import 'package:dart_ml/src/optimizer/initial_weights_generator/initial_weights_generator_factory.dart';
import 'package:dart_ml/src/optimizer/learning_rate_generator/generator_factory.dart';

class LogisticRegressor extends LinearClassifier {
  LogisticRegressor({
    int iterationLimit,
    double learningRate,
    double minWeightsUpdate,
    double lambda,
    int batchSize = 1,
    int randomSeed,
    bool fitIntercept = false,
    double interceptScale = 1.0,
    MultinomialType multinomialType = MultinomialType.oneVsAll,
    LearningRateType learningRateType = LearningRateType.decreasing
  }) : super(
    GradientOptimizer(
      RandomizerFactory.Default(randomSeed),
      CostFunctionFactory.LogLikelihood(),
      LearningRateGeneratorFactory.createByType(learningRateType),
      InitialWeightsGeneratorFactory.ZeroWeights(),

      initialLearningRate: learningRate,
      minCoefficientsUpdate: minWeightsUpdate,
      iterationLimit: iterationLimit,
      lambda: lambda,
      batchSize: batchSize
    ),
    scoreToProbabilityLink.logitLink,
    fitIntercept ? interceptScale : 0.0
  );
}
