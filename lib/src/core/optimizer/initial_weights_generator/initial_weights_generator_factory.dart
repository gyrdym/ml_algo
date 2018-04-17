import 'package:dart_ml/src/core/optimizer/initial_weights_generator/initial_weights_generator.dart';
import 'package:dart_ml/src/core/optimizer/initial_weights_generator/zero_weights_generator.dart';

class InitialWeightsGeneratorFactory {
  static InitialWeightsGenerator createZeroWeightsGenerator() => new ZeroWeightsGenerator();
}
