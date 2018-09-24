import 'dart:typed_data';

import 'package:dart_ml/src/optimizer/initial_weights_generator/initial_weights_generator.dart';
import 'package:dart_ml/src/optimizer/initial_weights_generator/zero_weights_generator.dart';

class InitialWeightsGeneratorFactory {
  static InitialWeightsGenerator<Float32x4List, Float32List, Float32x4> zeroWeights() => ZeroWeightsGenerator();
}
