import 'optimizer_type.dart';
import 'optimizer.dart';
import 'gradient/batch_optimizer.dart';
import 'gradient/mini_batch_optimizer.dart';
import 'gradient/stochastic_optimizer.dart';

class OptimizerFactory {
  static Optimizer create(OptimizerType type, [double learningRate = 1e-5, double minWeightsDistance = 1e-8, int iterationLimit = 10000]) {
    switch (type) {
      case OptimizerType.MBGD:
        return new MBGDOptimizer(learningRate, minWeightsDistance, iterationLimit);

      case OptimizerType.BGD:
        return new BGDOptimizer(learningRate, minWeightsDistance, iterationLimit);

      case OptimizerType.SGD:
        return new SGDOptimizer(learningRate, minWeightsDistance, iterationLimit);

      default:
        throw new UnsupportedError('Gradient descent type $type is not supported!');
    }
  }
}
