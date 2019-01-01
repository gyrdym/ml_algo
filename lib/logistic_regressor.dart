import 'package:ml_algo/src/classifier/float32x4_linear_classifier.dart';
import 'package:ml_algo/multinomial_type.dart';
import 'package:ml_algo/src/cost_function/cost_function_factory.dart';
import 'package:ml_algo/src/math/randomizer/randomizer_factory.dart';
import 'package:ml_algo/src/optimizer/gradient.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_generator_factory.dart';
import 'package:ml_algo/src/optimizer/learning_rate_generator/generator_factory.dart';
import 'package:ml_algo/learning_rate_type.dart';
import 'package:ml_algo/src/score_to_prob_link_function/float32x4_link_function.dart' as scoreToProbabilityLink;

class LogisticRegressor extends Float32x4LinearClassifier {
  LogisticRegressor(
      {int iterationLimit,
      double learningRate,
      double minWeightsUpdate,
      double lambda,
      int batchSize = 1,
      int randomSeed,
      bool fitIntercept = false,
      double interceptScale = 1.0,
      MultinomialType multinomialType = MultinomialType.oneVsAll,
      LearningRateType learningRateType = LearningRateType.decreasing})
      : super(
            GradientOptimizer(
                RandomizerFactory.defaultRandomizer(randomSeed),
                CostFunctionFactory.logLikelihood(scoreToProbabilityLink.vectorizedLogitLink),
                LearningRateGeneratorFactory.createByType(learningRateType),
                InitialWeightsGeneratorFactory.zeroWeights(),
                initialLearningRate: learningRate,
                minCoefficientsUpdate: minWeightsUpdate,
                iterationLimit: iterationLimit,
                lambda: lambda,
                batchSize: batchSize),
            scoreToProbabilityLink.vectorizedLogitLink,
            fitIntercept ? interceptScale : 0.0);
}
