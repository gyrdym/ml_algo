import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_generator.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_generator_factory.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/zero_weights_generator.dart';

class InitialWeightsGeneratorFactoryImpl
    implements InitialWeightsGeneratorFactory {
  const InitialWeightsGeneratorFactoryImpl();

  @override
  InitialWeightsGenerator zeroes() => ZeroWeightsGenerator();

  @override
  InitialWeightsGenerator fromType(InitialWeightsType type) {
    switch (type) {
      case InitialWeightsType.zeroes:
        return zeroes();
      default:
        throw UnimplementedError();
    }
  }
}
