import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_generator.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/zero_weights_generator.dart';

class InitialWeightsGeneratorFactory {
  static InitialWeightsGenerator<T> zeroWeights<T>() => ZeroWeightsGenerator<T>();
}
