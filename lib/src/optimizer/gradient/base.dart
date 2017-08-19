part of 'package:dart_ml/src/dart_ml.dart';

abstract class GradientOptimizer implements Optimizer {
  void configure({double learningRate, double minWeightsDistance, int iterationLimit, Regularization regularization,
                 LossFunction lossFunction, ScoreFunction scoreFunction,
                 double alpha = .00001, double argumentIncrement = 1e-5});
}