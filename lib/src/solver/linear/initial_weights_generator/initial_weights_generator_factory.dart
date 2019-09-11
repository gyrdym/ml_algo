import 'package:ml_algo/src/solver/linear/initial_weights_generator/initial_weights_generator.dart';
import 'package:ml_algo/src/solver/linear/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_linalg/dtype.dart';

abstract class InitialWeightsGeneratorFactory {
  InitialWeightsGenerator zeroes(DType dtype);
  InitialWeightsGenerator fromType(InitialWeightsType type, DType dtype);
}
