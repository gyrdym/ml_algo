import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_generator.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/linalg.dart';

class ZeroCoefficientsGenerator implements InitialCoefficientsGenerator {
  ZeroCoefficientsGenerator(this.dtype);

  final DType dtype;

  @override
  Vector generate(int length) => Vector.zero(length, dtype: dtype);
}
