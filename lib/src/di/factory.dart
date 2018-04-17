import 'package:dart_ml/src/core/data_splitter/factory.dart';
import 'package:dart_ml/src/core/data_splitter/splitter.dart';
import 'package:dart_ml/src/core/data_splitter/type.dart';
import 'package:dart_ml/src/core/loss_function/loss_function.dart';
import 'package:dart_ml/src/core/loss_function/loss_function_factory.dart';
import 'package:dart_ml/src/core/math/math.dart';
import 'package:dart_ml/src/core/math/math_analysis/gradient_calculator.dart';
import 'package:dart_ml/src/core/math/randomizer/randomizer.dart';
import 'package:dart_ml/src/core/metric/classification/metric_factory.dart';
import 'package:dart_ml/src/core/metric/classification/type.dart';
import 'package:dart_ml/src/core/metric/metric.dart';
import 'package:dart_ml/src/core/metric/regression/metric_factory.dart';
import 'package:dart_ml/src/core/metric/regression/type.dart';
import 'package:dart_ml/src/core/optimizer/gradient/factory.dart';
import 'package:dart_ml/src/core/optimizer/gradient/learning_rate_generator/learning_rate_generator.dart';
import 'package:dart_ml/src/core/optimizer/gradient/learning_rate_generator/learning_rate_generator_factory.dart';
import 'package:dart_ml/src/core/optimizer/initial_weights_generator/initial_weights_generator.dart';
import 'package:dart_ml/src/core/optimizer/initial_weights_generator/initial_weights_generator_factory.dart';
import 'package:dart_ml/src/core/optimizer/optimizer.dart';
import 'package:dart_ml/src/core/score_function/score_function.dart';
import 'package:dart_ml/src/core/score_function/score_function_factory.dart';
import 'package:di/di.dart';

class ModuleFactory {
  static Module modelSelectionModule(
    int value,
    {SplitterType splitter}
  ) => new Module()
      ..bind(Splitter, toFactory: () => splitter == null ? DataSplitterFactory.KFold(value) :
                                        DataSplitterFactory.createByType(splitter, value));

  static Module logisticRegressionModule({
    double learningRate,
    double minWeightsDistance,
    int iterationLimit,
    ClassificationMetricType metricType,
    double lambda,
    double argumentIncrement
  }) {
    return new Module()
      ..bind(Metric, toValue: metricType == null ? ClassificationMetricFactory.Accuracy() :
                              ClassificationMetricFactory.createByType(metricType))

      ..bind(LossFunction, toFactory: () => LossFunctionFactory.Logistic())
      ..bind(ScoreFunction, toFactory: () => ScoreFunctionFactory.Linear())
      ..bind(InitialWeightsGenerator, toFactory: () => InitialWeightsGeneratorFactory.createZeroWeightsGenerator())
      ..bind(GradientCalculator, toFactory: () => MathUtils.createGradientCalculator())
      ..bind(LearningRateGenerator, toFactory: () => LearningRateGeneratorFactory.createSimpleGenerator())
      ..bind(Optimizer, toFactory: () => gradientOptimizerFactory(learningRate, minWeightsDistance, iterationLimit,
          lambda, argumentIncrement, 1))
      ..bind(Randomizer, toFactory: () => MathUtils.createRandomizer());
  }

  static Module GradientRegressionModule({
    double learningRate,
    double minWeightsDistance,
    int iterationLimit,
    RegressionMetricType metric,
    double lambda,
    double argumentIncrement,
    int batchSize = 1
  }) {
    return new Module()
      ..bind(Metric, toValue: metric == null ? RegressionMetricFactory.RMSE() :
                              RegressionMetricFactory.createByType(metric))

      ..bind(LossFunction, toFactory: () => LossFunctionFactory.Squared())
      ..bind(ScoreFunction, toFactory: () => ScoreFunctionFactory.Linear())
      ..bind(InitialWeightsGenerator, toFactory: () => InitialWeightsGeneratorFactory.createZeroWeightsGenerator())
      ..bind(GradientCalculator, toFactory: () => MathUtils.createGradientCalculator())
      ..bind(LearningRateGenerator, toFactory: () => LearningRateGeneratorFactory.createSimpleGenerator())
      ..bind(Optimizer, toFactory: () => gradientOptimizerFactory(learningRate, minWeightsDistance, iterationLimit,
          lambda, argumentIncrement, batchSize))
      ..bind(Randomizer, toFactory: () => MathUtils.createRandomizer());
  }
}