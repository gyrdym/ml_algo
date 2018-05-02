import 'package:dart_ml/src/optimizer/initial_weights_generator/initial_weights_generator.dart';
import 'package:simd_vector/vector.dart';

class ZeroWeightsGenerator implements InitialWeightsGenerator {
  Float64x2Vector generate(int length) => new Float64x2Vector.from(new List.filled(length, 0.0));
}
