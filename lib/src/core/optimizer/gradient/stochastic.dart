import 'package:dart_ml/src/core/math/randomizer/randomizer.dart';
import 'package:dart_ml/src/core/optimizer/gradient/optimizer.dart';
import 'package:dart_ml/src/core/optimizer/regularization.dart';
import 'package:dart_ml/src/di/injector.dart';

class SGDOptimizerImpl extends GradientOptimizerImpl {
  final Randomizer _randomizer = coreInjector.get(Randomizer);

  SGDOptimizerImpl({
    double learningRate,
    double minWeightsDistance,
    int iterationLimit,
    Regularization regularization,
    double alpha,
    double argumentIncrement
  }) : super(
    learningRate: learningRate,
    minWeightsDiff: minWeightsDistance,
    iterationLimit: iterationLimit,
    lambda: alpha,
    argumentIncrement: argumentIncrement
  );

  @override
  Iterable<int> getBatchRange(int numberOfPoints) {
    int k = _randomizer.getIntegerFromInterval(0, numberOfPoints);
    return [k, k + 1];
  }
}
