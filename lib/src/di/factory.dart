part of 'package:dart_ml/src/core/implementation.dart';

class ModuleFactory {
  static Module modelSelectionModule(
    int value,
    {SplitterType splitter}
  ) => new Module()
      ..bind(
        Splitter,
        toFactory: () => splitter == null ?
                         DataSplitterFactory.KFold(value) : DataSplitterFactory.createByType(splitter, value));

  static Module logisticRegressionModule({
    double learningRate,
    double minWeightsDistance,
    int iterationLimit,
    ClassificationMetricType metricType,
    Regularization regularization,
    double lambda,
    double argumentIncrement
  }) {
    return new Module()
      ..bind(Metric, toValue: metricType == null ? ClassificationMetricFactory.Accuracy() :
                              ClassificationMetricFactory.createByType(metricType))

      ..bind(LossFunction, toFactory: () => LossFunctionFactory.LogisticLoss())
      ..bind(ScoreFunction, toFactory: () => ScoreFunctionFactory.Linear())
      ..bind(InitialWeightsGenerator, toFactory: () => InitialWeightsGeneratorFactory.createZeroWeightsGenerator())
      ..bind(GradientCalculator, toFactory: () => MathUtils.createGradientCalculator())
      ..bind(LearningRateGenerator, toFactory: () => LearningRateGeneratorFactory.createSimpleGenerator())
      ..bind(Optimizer, toFactory: () => GradientOptimizerFactory.createStochasticOptimizer(
        learningRate, minWeightsDistance, iterationLimit, regularization, lambda, argumentIncrement
      ))
      ..bind(Randomizer, toFactory: () => MathUtils.createRandomizer());
  }

  static Module SGDRegressionModule({
    double learningRate,
    double minWeightsDistance,
    int iterationLimit,
    RegressionMetricType metric,
    Regularization regularization, alpha,
    double argumentIncrement
  }) {
    return new Module()
      ..bind(Metric, toValue: metric == null ? RegressionMetricFactory.RMSE() :
                              RegressionMetricFactory.createByType(metric))

      ..bind(LossFunction, toFactory: () => LossFunctionFactory.Squared())
      ..bind(ScoreFunction, toFactory: () => ScoreFunctionFactory.Linear())
      ..bind(InitialWeightsGenerator, toFactory: () => InitialWeightsGeneratorFactory.createZeroWeightsGenerator())
      ..bind(GradientCalculator, toFactory: () => MathUtils.createGradientCalculator())
      ..bind(LearningRateGenerator, toFactory: () => LearningRateGeneratorFactory.createSimpleGenerator())
      ..bind(Optimizer, toFactory: () => GradientOptimizerFactory.createStochasticOptimizer(
          learningRate, minWeightsDistance, iterationLimit, regularization, alpha, argumentIncrement
        ))
      ..bind(Randomizer, toFactory: () => MathUtils.createRandomizer());
  }

  static Module MBGDRegressionModule({
    double learningRate,
    double minWeightsDistance,
    int iterationLimit,
    RegressionMetricType metric,
    Regularization regularization,
    double lambda,
    double argumentIncrement
  }) {
    return new Module()
      ..bind(Metric, toValue: metric == null ? RegressionMetricFactory.RMSE() :
                              RegressionMetricFactory.createByType(metric))

      ..bind(LossFunction, toFactory: () => LossFunctionFactory.Squared())
      ..bind(ScoreFunction, toFactory: () => ScoreFunctionFactory.Linear())
      ..bind(InitialWeightsGenerator, toFactory: () => InitialWeightsGeneratorFactory.createZeroWeightsGenerator())
      ..bind(GradientCalculator, toFactory: () => MathUtils.createGradientCalculator())
      ..bind(LearningRateGenerator, toFactory: () => LearningRateGeneratorFactory.createSimpleGenerator())
      ..bind(Optimizer, toFactory: () => GradientOptimizerFactory.createMiniBatchOptimizer(learningRate,
          minWeightsDistance, iterationLimit, regularization, lambda, argumentIncrement))
      ..bind(Randomizer, toFactory: () => MathUtils.createRandomizer());
  }

  static Module BGDRegressionModule({
    double learningRate,
    double minWeightsDistance,
    int iterationLimit,
    RegressionMetricType metric,
    Regularization regularization,
    double lambda,
    double argumentIncrement
  }) {
    return new Module()
      ..bind(Metric, toValue: metric == null ? RegressionMetricFactory.RMSE() :
                              RegressionMetricFactory.createByType(metric))

      ..bind(LossFunction, toFactory: () => LossFunctionFactory.Squared())
      ..bind(ScoreFunction, toFactory: () => ScoreFunctionFactory.Linear())
      ..bind(InitialWeightsGenerator, toFactory: () => InitialWeightsGeneratorFactory.createZeroWeightsGenerator())
      ..bind(GradientCalculator, toFactory: () => MathUtils.createGradientCalculator())
      ..bind(LearningRateGenerator, toFactory: () => LearningRateGeneratorFactory.createSimpleGenerator())
      ..bind(Optimizer, toFactory: () => GradientOptimizerFactory.createBatchOptimizer(learningRate,
          minWeightsDistance, iterationLimit, regularization, lambda, argumentIncrement))
      ..bind(Randomizer, toFactory: () => MathUtils.createRandomizer());
  }
}