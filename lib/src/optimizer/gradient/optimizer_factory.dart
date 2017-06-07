import 'package:dart_ml/src/di/injector.dart';
import 'package:dart_ml/src/optimizer/regularization.dart';

import 'optimizer.dart';
import 'optimizer_type.dart';

import 'interface/batch.dart';
import 'interface/mini_batch.dart';
import 'interface/stochastic.dart';

class GradientOptimizerFactory {
  static GradientOptimizer create(GradientOptimizerType type, double learningRate, double minWeightsDistance,
                                  int iterationLimit, Regularization regularization) {
    GradientOptimizer optimizer;

    switch (type) {
      case GradientOptimizerType.MBGD:
        optimizer = injector.get(MBGDOptimizer);
        break;

      case GradientOptimizerType.BGD:
        optimizer = injector.get(BGDOptimizer);
        break;

      case GradientOptimizerType.SGD:
        optimizer = injector.get(SGDOptimizer);
        break;

      default:
        throw new UnsupportedError('Gradient descent type $type is not supported!');
    }

    return optimizer
      ..configure(learningRate, minWeightsDistance, iterationLimit, regularization);
  }
}
