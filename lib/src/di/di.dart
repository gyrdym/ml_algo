import 'package:di/di.dart';
import 'package:dart_ml/src/di/injector.dart' show injector;

import 'package:dart_ml/src/optimizer/gradient/interface/batch.dart';
import 'package:dart_ml/src/optimizer/gradient/interface/mini_batch.dart';
import 'package:dart_ml/src/optimizer/gradient/interface/stochastic.dart';

import 'package:dart_ml/src/optimizer/gradient/implementation/batch.dart';
import 'package:dart_ml/src/optimizer/gradient/implementation/mini_batch.dart';
import 'package:dart_ml/src/optimizer/gradient/implementation/stochastic.dart';

class DiConfigurator {
  static void configure() {
    injector = new ModuleInjector([new Module()
      ..bind(BGDOptimizer, toFactory: () => new BGDOptimizerImpl())
      ..bind(MBGDOptimizer, toFactory: () => new MBGDOptimizerImpl())
      ..bind(SGDOptimizer, toFactory: () => new SGDOptimizerImpl())
    ]);
  }
}