part of 'package:dart_ml/src/core/implementation.dart';

class GradientOptimizerFactory {
  static Optimizer createBatchOptimizer(
    double learningRate,
    double minWeightsDistance,
    int iterationLimit,
    Regularization regularization,
    double lambda,
    double argumentIncrement
  ) =>
    new _BGDOptimizerImpl(
      learningRate: learningRate,
      minWeightsDistance: minWeightsDistance,
      iterationLimit: iterationLimit,
      regularization: regularization,
      lambda: lambda,
      argumentIncrement: argumentIncrement
    );

  static Optimizer createMiniBatchOptimizer(
    double learningRate,
    double minWeightsDistance,
    int iterationLimit,
    Regularization regularization,
    double lambda,
    double argumentIncrement
  ) =>
    new _MBGDOptimizerImpl(
      learningRate: learningRate,
      minWeightsDistance: minWeightsDistance,
      iterationLimit: iterationLimit,
      regularization: regularization,
      alpha: lambda,
      argumentIncrement: argumentIncrement
    );

  static Optimizer createStochasticOptimizer(
    double learningRate,
    double minWeightsDistance,
    int iterationLimit,
    Regularization regularization,
    double lambda,
    double argumentIncrement
  ) =>
    new _SGDOptimizerImpl(
      learningRate: learningRate,
      minWeightsDistance: minWeightsDistance,
      iterationLimit: iterationLimit,
      regularization: regularization,
      alpha: lambda,
      argumentIncrement: argumentIncrement
    );
}
