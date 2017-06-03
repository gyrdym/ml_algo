import 'package:dart_ml/src/optimizer/regularization.dart';

import 'optimizer.dart';
import 'optimizer_type.dart';
import 'batch.dart';
import 'mini_batch.dart';
import 'stochastic.dart';

class GradientOptimizerFactory {
  static GradientOptimizer create(GradientOptimizerType type, double learningRate, double minWeightsDistance,
                                  int iterationLimit, Regularization regularization) {
    switch (type) {
      case GradientOptimizerType.MBGD:
        return new MBGDOptimizer(learningRate, minWeightsDistance, iterationLimit, regularization);

      case GradientOptimizerType.BGD:
        return new BGDOptimizer(learningRate, minWeightsDistance, iterationLimit, regularization);

      case GradientOptimizerType.SGD:
        return new SGDOptimizer(learningRate, minWeightsDistance, iterationLimit, regularization);

      default:
        throw new UnsupportedError('Gradient descent type $type is not supported!');
    }
  }
}
