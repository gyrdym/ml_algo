import 'package:dart_ml/src/core/optimizer/gradient/optimizer.dart';
import 'package:dart_ml/src/core/optimizer/regularization.dart';

class BGDOptimizerImpl extends GradientOptimizerImpl {

  BGDOptimizerImpl({
    double learningRate,
    double minWeightsDistance,
    int iterationLimit,
    Regularization regularization,
    double lambda,
    double argumentIncrement
  }) : super(
    learningRate: learningRate,
    minWeightsDiff: minWeightsDistance,
    iterationLimit: iterationLimit,
    lambda: lambda,
    argumentIncrement: argumentIncrement
  );

  @override
  Iterable<int> getBatchRange(int numberOfPoints) => [0, numberOfPoints];
}
