import 'package:dart_ml/src/optimizer/regularization/regularization.dart';
import 'package:dart_ml/src/optimizer/interface/optimizer.dart';

abstract class GradientOptimizer implements Optimizer {
  void configure(double learningRate, double minWeightsDistance, int iterationLimit, Regularization regularization,
                 {double alpha = .00001});
}