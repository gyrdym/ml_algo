import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_generator.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_generator_factory.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/zero_coefficients_generator.dart';
import 'package:ml_linalg/dtype.dart';

class InitialCoefficientsGeneratorFactoryImpl
    implements InitialCoefficientsGeneratorFactory {
  const InitialCoefficientsGeneratorFactoryImpl();

  @override
  InitialCoefficientsGenerator zeroes(DType dtype) => ZeroCoefficientsGenerator(dtype);

  @override
  InitialCoefficientsGenerator fromType(InitialCoefficientsType type, DType dtype) {
    switch (type) {
      case InitialCoefficientsType.zeroes:
        return zeroes(dtype);
      default:
        throw UnsupportedError('Unsupported initial weights type - $type');
    }
  }
}
