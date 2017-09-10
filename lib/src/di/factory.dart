import 'package:di/di.dart';

import 'package:dart_ml/src/interface.dart';
import 'package:dart_ml/src/implementation.dart';

class InjectorFactory {
  static ModuleInjector create() {
    return new ModuleInjector([new Module()
      ..bind(DerivativeFinder, toFactory: () => MathUtils.createDerivativeFinder())
      ..bind(Randomizer, toFactory: () => MathUtils.createRandomizer())
      ..bind(BGDOptimizer, toFactory: () => GradientOptimizerFactory.createBatchOptimizer())
      ..bind(MBGDOptimizer, toFactory: () => GradientOptimizerFactory.createMiniBatchOptimizer())
      ..bind(SGDOptimizer, toFactory: () => GradientOptimizerFactory.createStochasticOptimizer())
      ..bind(KFoldSplitter, toFactory: () => DataSplitterFactory.createKFoldSplitter())
      ..bind(LeavePOutSplitter, toFactory: () => DataSplitterFactory.createLpoSplitter())
    ]);
  }
}