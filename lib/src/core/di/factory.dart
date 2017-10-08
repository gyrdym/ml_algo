part of 'package:dart_ml/src/core/implementation.dart';

class ModuleFactory {
  static Module createLogisticRegressionModule({double learningRate, double minWeightsDistance, int iterationLimit,
    ClassificationMetric metric, Regularization regularization, alpha, double argumentIncrement}) {

    return new Module()
      ..bind(Metric, toValue: metric ?? ClassificationMetricFactory.Accuracy())
      ..bind(LossFunction, toFactory: () => LossFunctionFactory.LogisticLoss())
      ..bind(ScoreFunction, toFactory: () => ScoreFunctionFactory.Linear())
      ..bind(InitialWeightsGenerator, toFactory: () => InitialWeightsGeneratorFactory.createZeroWeightsGenerator())
      ..bind(GradientCalculator, toFactory: () => MathUtils.createGradientCalculator())
      ..bind(LearningRateGenerator, toFactory: () => LearningRateGeneratorFactory.createSimpleGenerator())
      ..bind(Optimizer, toFactory: () => GradientOptimizerFactory.createStochasticOptimizer(
        learningRate, minWeightsDistance, iterationLimit, regularization, alpha, argumentIncrement
      ))
      ..bind(KFoldSplitter, toFactory: () => DataSplitterFactory.createKFoldSplitter())
      ..bind(Randomizer, toFactory: () => MathUtils.createRandomizer());
  }

  static Module createSGDRegressionModule({double learningRate, double minWeightsDistance, int iterationLimit,
    RegressionMetric metric, Regularization regularization, alpha, double argumentIncrement}) {

    return new Module()
      ..bind(Metric, toValue: metric ?? RegressionMetricFactory.RMSE())
      ..bind(LossFunction, toFactory: () => LossFunctionFactory.Squared())
      ..bind(ScoreFunction, toFactory: () => ScoreFunctionFactory.Linear())
      ..bind(InitialWeightsGenerator, toFactory: () => InitialWeightsGeneratorFactory.createZeroWeightsGenerator())
      ..bind(GradientCalculator, toFactory: () => MathUtils.createGradientCalculator())
      ..bind(LearningRateGenerator, toFactory: () => LearningRateGeneratorFactory.createSimpleGenerator())
      ..bind(Optimizer, toFactory: () => GradientOptimizerFactory.createStochasticOptimizer(
          learningRate, minWeightsDistance, iterationLimit, regularization, alpha, argumentIncrement
          ))
      ..bind(KFoldSplitter, toFactory: () => DataSplitterFactory.createKFoldSplitter())
      ..bind(Randomizer, toFactory: () => MathUtils.createRandomizer());
  }

  static Module createMBGDRegressionModule({double learningRate, double minWeightsDistance, int iterationLimit,
    RegressionMetric metric, Regularization regularization, alpha, double argumentIncrement}) {

    return new Module()
      ..bind(Metric, toValue: metric)
      ..bind(LossFunction, toFactory: () => LossFunctionFactory.Squared())
      ..bind(ScoreFunction, toFactory: () => ScoreFunctionFactory.Linear())
      ..bind(InitialWeightsGenerator, toFactory: () => InitialWeightsGeneratorFactory.createZeroWeightsGenerator())
      ..bind(GradientCalculator, toFactory: () => MathUtils.createGradientCalculator())
      ..bind(LearningRateGenerator, toFactory: () => LearningRateGeneratorFactory.createSimpleGenerator())
      ..bind(Optimizer, toFactory: () => GradientOptimizerFactory.createMiniBatchOptimizer(learningRate,
          minWeightsDistance, iterationLimit, regularization, alpha, argumentIncrement))
      ..bind(KFoldSplitter, toFactory: () => DataSplitterFactory.createKFoldSplitter())
      ..bind(Randomizer, toFactory: () => MathUtils.createRandomizer());
  }

  static Module createBGDRegressionModule({double learningRate, double minWeightsDistance, int iterationLimit,
    RegressionMetric metric, Regularization regularization, alpha, double argumentIncrement}) {

    return new Module()
      ..bind(Metric, toValue: metric)
      ..bind(LossFunction, toFactory: () => LossFunctionFactory.Squared())
      ..bind(ScoreFunction, toFactory: () => ScoreFunctionFactory.Linear())
      ..bind(InitialWeightsGenerator, toFactory: () => InitialWeightsGeneratorFactory.createZeroWeightsGenerator())
      ..bind(GradientCalculator, toFactory: () => MathUtils.createGradientCalculator())
      ..bind(LearningRateGenerator, toFactory: () => LearningRateGeneratorFactory.createSimpleGenerator())
      ..bind(Optimizer, toFactory: () => GradientOptimizerFactory.createBatchOptimizer(learningRate,
          minWeightsDistance, iterationLimit, regularization, alpha, argumentIncrement))
      ..bind(KFoldSplitter, toFactory: () => DataSplitterFactory.createKFoldSplitter())
      ..bind(Randomizer, toFactory: () => MathUtils.createRandomizer());
  }
}