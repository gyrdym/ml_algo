part of 'package:dart_ml/src/core/implementation.dart';

Module createLassoRegressionModule({
  double minWeightsDistance,
  int iterationLimit,
  double lambda,
  Metric metric,
  ScoreFunction scoreFn
}) {
  return new Module()
    ..bind(Metric, toValue: metric ?? RegressionMetricFactory.RMSE())
    ..bind(ScoreFunction, toFactory: () => scoreFn ?? ScoreFunctionFactory.Linear())
    ..bind(InitialWeightsGenerator, toFactory: () => InitialWeightsGeneratorFactory.createZeroWeightsGenerator())
    ..bind(Optimizer, toFactory: () => CoordinateOptimizerFactory
        .createCoordinateOptimizer(minWeightsDistance, iterationLimit, lambda))
    ..bind(Randomizer, toFactory: () => MathUtils.createRandomizer());
}