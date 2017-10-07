import 'package:di/di.dart';

import 'package:dart_ml/src/loss_function/loss_function.dart';
import 'package:dart_ml/src/score_function/score_function.dart';
import 'package:dart_ml/src/interface.dart';
import 'package:dart_ml/src/implementation.dart';

class ModuleFactory {
  static Module createLogisticRegressionModule({double learningRate, double minWeightsDistance, int iterationLimit,
                                                 Regularization regularization, alpha, double argumentIncrement}) {

    return new Module()
      ..bind(ClassificationMetric, toFactory: () => ClassificationMetricFactory.Accuracy())
      ..bind(LossFunction, toFactory: () => new LossFunction.LogisticLoss())
      ..bind(ScoreFunction, toFactory: () => new ScoreFunction.Linear())
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
    Metric metric, Regularization regularization, alpha, double argumentIncrement}) {

    return new Module()
      ..bind(Metric, toValue: metric)
      ..bind(LossFunction, toFactory: () => new LossFunction.LogisticLoss())
      ..bind(ScoreFunction, toFactory: () => new ScoreFunction.Linear())
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
    Metric metric, Regularization regularization, alpha, double argumentIncrement}) {

    return new Module()
      ..bind(Metric, toValue: metric)
      ..bind(LossFunction, toFactory: () => new LossFunction.LogisticLoss())
      ..bind(ScoreFunction, toFactory: () => new ScoreFunction.Linear())
      ..bind(InitialWeightsGenerator, toFactory: () => InitialWeightsGeneratorFactory.createZeroWeightsGenerator())
      ..bind(GradientCalculator, toFactory: () => MathUtils.createGradientCalculator())
      ..bind(LearningRateGenerator, toFactory: () => LearningRateGeneratorFactory.createSimpleGenerator())
      ..bind(Optimizer, toFactory: () => GradientOptimizerFactory.createMiniBatchOptimizer(learningRate,
          minWeightsDistance, iterationLimit, regularization, alpha, argumentIncrement))
      ..bind(KFoldSplitter, toFactory: () => DataSplitterFactory.createKFoldSplitter())
      ..bind(Randomizer, toFactory: () => MathUtils.createRandomizer());
  }

  static Module createBGDRegressionModule({double learningRate, double minWeightsDistance, int iterationLimit,
                                             Metric metric, Regularization regularization, alpha, double argumentIncrement}) {

    return new Module()
      ..bind(Metric, toValue: metric)
      ..bind(LossFunction, toFactory: () => new LossFunction.LogisticLoss())
      ..bind(ScoreFunction, toFactory: () => new ScoreFunction.Linear())
      ..bind(InitialWeightsGenerator, toFactory: () => InitialWeightsGeneratorFactory.createZeroWeightsGenerator())
      ..bind(GradientCalculator, toFactory: () => MathUtils.createGradientCalculator())
      ..bind(LearningRateGenerator, toFactory: () => LearningRateGeneratorFactory.createSimpleGenerator())
      ..bind(Optimizer, toFactory: () => GradientOptimizerFactory.createBatchOptimizer(learningRate,
          minWeightsDistance, iterationLimit, regularization, alpha, argumentIncrement))
      ..bind(KFoldSplitter, toFactory: () => DataSplitterFactory.createKFoldSplitter())
      ..bind(Randomizer, toFactory: () => MathUtils.createRandomizer());
  }
}