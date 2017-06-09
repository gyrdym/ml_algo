import 'package:di/di.dart';
import 'package:dart_ml/src/di/injector.dart' show injector;

import 'package:dart_ml/src/math/misc/randomizer/interface/randomizer.dart';
import 'package:dart_ml/src/math/misc/randomizer/implementation/randomizer.dart';

import 'package:dart_ml/src/optimizer/optimizer.dart';
import 'package:dart_ml/src/optimizer/optimizer_impl.dart';

import 'package:dart_ml/src/data_splitter/data_splitter.dart';
import 'package:dart_ml/src/data_splitter/data_splitter_impl.dart';

class DiConfigurator {
  static void configure() {
    injector = new ModuleInjector([new Module()
      ..bind(Randomizer, toFactory: () => new RandomizerImpl())
      ..bind(BGDOptimizer, toFactory: () => new BGDOptimizerImpl())
      ..bind(MBGDOptimizer, toFactory: () => new MBGDOptimizerImpl())
      ..bind(SGDOptimizer, toFactory: () => new SGDOptimizerImpl())
      ..bind(KFoldSplitter, toFactory: () => new KFoldSplitterImpl())
      ..bind(LeavePOutSplitter, toFactory: () => new LeavePOutSplitterImpl())
    ]);
  }
}