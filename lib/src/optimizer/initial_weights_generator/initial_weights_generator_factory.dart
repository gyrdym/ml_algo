import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_generator.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_type.dart';

abstract class InitialWeightsGeneratorFactory {
  InitialWeightsGenerator zeroes();
  InitialWeightsGenerator fromType(InitialWeightsType type);
}