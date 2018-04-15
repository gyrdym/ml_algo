import 'package:dart_ml/src/core/optimizer/gradient/batch.dart';
import 'package:dart_ml/src/core/optimizer/gradient/mini_batch.dart';
import 'package:dart_ml/src/core/optimizer/gradient/stochastic.dart';
import 'package:dart_ml/src/core/optimizer/optimizer.dart';
import 'package:dart_ml/src/core/optimizer/regularization.dart';

class GradientOptimizerFactory {
  static Optimizer createBatchOptimizer(
    double learningRate,
    double minWeightsDistance,
    int iterationLimit,
    Regularization regularization,
    double lambda,
    double argumentIncrement
  ) =>
    new BGDOptimizerImpl(
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
    new MBGDOptimizerImpl(
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
    new SGDOptimizerImpl(
      learningRate: learningRate,
      minWeightsDistance: minWeightsDistance,
      iterationLimit: iterationLimit,
      regularization: regularization,
      alpha: lambda,
      argumentIncrement: argumentIncrement
    );
}
