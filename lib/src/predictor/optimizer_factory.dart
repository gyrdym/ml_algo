part of linear_regressor;

class _OptimizerFactory {
  static Optimizer create(GradientDescent type, [double learningRate = 1e-5, double minWeightsDistance = 1e-8, int iterationLimit = 10000]) {
    switch (type) {
      case GradientDescent.MBGD:
        return new MBGDOptimizer(learningRate, minWeightsDistance, iterationLimit);

      case GradientDescent.BGD:
        return new BGDOptimizer(learningRate, minWeightsDistance, iterationLimit);

      case GradientDescent.SGD:
        return new SGDOptimizer(learningRate, minWeightsDistance, iterationLimit);

      default:
        throw new UnsupportedError('Gradient descent type $type is not supported!');
    }
  }
}
