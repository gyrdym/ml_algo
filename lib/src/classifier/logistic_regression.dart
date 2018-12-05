import 'package:ml_algo/src/classifier/linear_classifier.dart';
import 'package:ml_algo/src/classifier/multinomial_type.dart';
import 'package:ml_algo/src/cost_function/cost_function_factory.dart';
import 'package:ml_algo/src/optimizer/learning_rate_generator/type.dart';
import 'package:ml_algo/src/score_to_prob_link_function/link_function_impl.dart' as scoreToProbabilityLink;
import 'package:ml_algo/src/math/randomizer/randomizer_factory.dart';
import 'package:ml_algo/src/optimizer/gradient.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_generator_factory.dart';
import 'package:ml_algo/src/optimizer/learning_rate_generator/generator_factory.dart';

class LogisticRegressor extends LinearClassifier {
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
            scoreToProbabilityLink.logitLink,
            fitIntercept ? interceptScale : 0.0);
}
