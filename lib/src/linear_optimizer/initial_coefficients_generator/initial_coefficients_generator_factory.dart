import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_generator.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_linalg/dtype.dart';

abstract class InitialCoefficientsGeneratorFactory {
  InitialCoefficientsGenerator zeroes(DType dtype);
  InitialCoefficientsGenerator fromType(InitialCoefficientsType type, DType dtype);
}
