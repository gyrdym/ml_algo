import 'package:dart_ml/src/core/math/math.dart';
import 'package:dart_ml/src/core/math/randomizer/randomizer.dart';
import 'package:dart_ml/src/core/metric/metric.dart';
import 'package:dart_ml/src/core/metric/regression/metric_factory.dart';
import 'package:dart_ml/src/core/optimizer/coordinate/factory.dart';
import 'package:dart_ml/src/core/optimizer/initial_weights_generator/initial_weights_generator.dart';
import 'package:dart_ml/src/core/optimizer/initial_weights_generator/initial_weights_generator_factory.dart';
import 'package:dart_ml/src/core/optimizer/optimizer.dart';
import 'package:dart_ml/src/core/score_function/score_function.dart';
import 'package:dart_ml/src/core/score_function/score_function_factory.dart';
import 'package:di/di.dart';

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