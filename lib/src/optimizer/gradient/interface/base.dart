import 'package:dart_ml/src/optimizer/regularization/regularization.dart';
import 'package:dart_ml/src/optimizer/interface/optimizer.dart';
import 'package:dart_ml/src/loss_function/loss_function.dart';

abstract class GradientOptimizer implements Optimizer {
  void configure(double learningRate, double minWeightsDistance, int iterationLimit, Regularization regularization,
                 LossFunction lossFunction, {double alpha = .00001, double argumentIncrement = 1e-5});
}