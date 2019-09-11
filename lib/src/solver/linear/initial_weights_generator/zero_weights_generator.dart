import 'package:ml_algo/src/solver/linear/initial_weights_generator/initial_weights_generator.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/linalg.dart';

class ZeroWeightsGenerator implements InitialWeightsGenerator {
  ZeroWeightsGenerator(this.dtype);

  final DType dtype;

  @override
  Vector generate(int length) => Vector.zero(length, dtype: dtype);
}
