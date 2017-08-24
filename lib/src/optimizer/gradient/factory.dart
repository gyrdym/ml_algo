part of 'package:dart_ml/src/implementation.dart';

class GradientOptimizerFactory {
  static BGDOptimizer createBatchOptimizer() => new _BGDOptimizerImpl();
  static MBGDOptimizer createMiniBatchOptimizer() => new _MBGDOptimizerImpl();
  static SGDOptimizer createStochasticOptimizer() => new _SGDOptimizerImpl();
}
