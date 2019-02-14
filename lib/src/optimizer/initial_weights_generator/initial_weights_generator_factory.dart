import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_generator.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_type.dart';

abstract class InitialWeightsGeneratorFactory {
  InitialWeightsGenerator zeroes(Type dtype);
  InitialWeightsGenerator fromType(InitialWeightsType type, Type dtype);
}
