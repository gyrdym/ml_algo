import 'package:dart_ml/src/optimizer/initial_weights_generator/initial_weights_generator.dart';
import 'package:linalg/vector.dart';

class ZeroWeightsGenerator implements InitialWeightsGenerator {
  Float32x4Vector generate(int length) => new Float32x4Vector.from(new List.filled(length, 0.0));
}
