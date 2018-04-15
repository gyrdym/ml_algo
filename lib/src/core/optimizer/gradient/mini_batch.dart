import 'package:dart_ml/src/core/math/randomizer/randomizer.dart';
import 'package:dart_ml/src/core/optimizer/gradient/optimizer.dart';
import 'package:dart_ml/src/core/optimizer/regularization.dart';
import 'package:dart_ml/src/di/injector.dart';

class MBGDOptimizerImpl extends GradientOptimizerImpl {
  final Randomizer _randomizer = coreInjector.get(Randomizer);

  MBGDOptimizerImpl({
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
  Iterable<int> getBatchRange(int numberOfPoints) => _randomizer.getIntegerInterval(0, numberOfPoints);
}
