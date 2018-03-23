part of 'package:dart_ml/src/core/implementation.dart';

class _BGDOptimizerImpl extends _GradientOptimizerImpl {

  _BGDOptimizerImpl({
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
  Iterable<int> _getBatchRange(int numberOfPoints) => [0, numberOfPoints];
}
