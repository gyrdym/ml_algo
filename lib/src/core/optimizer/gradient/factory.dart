part of 'package:dart_ml/src/core/implementation.dart';

class GradientOptimizerFactory {
  static BGDOptimizer createBatchOptimizer(
    double learningRate,
    double minWeightsDistance,
    int iterationLimit,
    Regularization regularization,
    double alpha,
    double argumentIncrement
  ) =>
    new _BGDOptimizerImpl(
      learningRate: learningRate,
      minWeightsDistance: minWeightsDistance,
      iterationLimit: iterationLimit,
      regularization: regularization,
      alpha: alpha,
      argumentIncrement: argumentIncrement
    );

  static MBGDOptimizer createMiniBatchOptimizer(
    double learningRate,
    double minWeightsDistance,
    int iterationLimit,
    Regularization regularization,
    double alpha,
    double argumentIncrement
  ) =>
    new _MBGDOptimizerImpl(
      learningRate: learningRate,
      minWeightsDistance: minWeightsDistance,
      iterationLimit: iterationLimit,
      regularization: regularization,
      alpha: alpha,
      argumentIncrement: argumentIncrement
    );

  static Optimizer createStochasticOptimizer(
    double learningRate,
    double minWeightsDistance,
    int iterationLimit,
    Regularization regularization,
    double alpha,
    double argumentIncrement
  ) =>
    new _SGDOptimizerImpl(
      learningRate: learningRate,
      minWeightsDistance: minWeightsDistance,
      iterationLimit: iterationLimit,
      regularization: regularization,
      alpha: alpha,
      argumentIncrement: argumentIncrement
    );
}
