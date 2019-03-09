import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_generator.dart';
import 'package:ml_linalg/linalg.dart';

class ZeroWeightsGenerator implements InitialWeightsGenerator {
  ZeroWeightsGenerator(this.dtype);

  final Type dtype;

  @override
  Vector generate(int length) => Vector.zero(length, dtype: dtype);
}
