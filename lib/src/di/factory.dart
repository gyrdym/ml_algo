import 'package:di/di.dart';

import 'package:dart_ml/src/dart_ml.dart';
import 'package:dart_ml/src/dart_ml_impl.dart';

import 'package:dart_ml/src/math/misc/randomizer/randomizer.dart';
import 'package:dart_ml/src/math/misc/randomizer/randomizer_impl.dart';

class InjectorFactory {
  static ModuleInjector create() {
    return new ModuleInjector([new Module()
      ..bind(Randomizer, toFactory: () => new RandomizerImpl())
      ..bind(BGDOptimizer, toFactory: () => GradientOptimizerFactory.createBatchOptimizer())
      ..bind(MBGDOptimizer, toFactory: () => GradientOptimizerFactory.createMiniBatchOptimizer())
      ..bind(SGDOptimizer, toFactory: () => GradientOptimizerFactory.createStochasticOptimizer())
      ..bind(KFoldSplitter, toFactory: () => DataSplitterFactory.createKFoldSplitter())
      ..bind(LeavePOutSplitter, toFactory: () => DataSplitterFactory.createLpoSplitter())
    ]);
  }
}