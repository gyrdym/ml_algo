import 'package:di/di.dart';

import 'package:dart_ml/src/dart_ml.dart';
import 'package:dart_ml/src/dart_ml_impl.dart';

import 'package:dart_ml/src/math/misc/randomizer/randomizer.dart';
import 'package:dart_ml/src/math/misc/randomizer/randomizer_impl.dart';

class InjectorFactory {
  static ModuleInjector create() {
    return new ModuleInjector([new Module()
      ..bind(Randomizer, toFactory: () => new RandomizerImpl())
      ..bind(BGDOptimizer, toFactory: () => new BGDOptimizerImpl())
      ..bind(MBGDOptimizer, toFactory: () => new MBGDOptimizerImpl())
      ..bind(SGDOptimizer, toFactory: () => new SGDOptimizerImpl())
      ..bind(KFoldSplitter, toFactory: () => DataSplitterFactory.KFold())
      ..bind(LeavePOutSplitter, toFactory: () => DataSplitterFactory.Lpo())
    ]);
  }
}