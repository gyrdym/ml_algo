import 'package:dart_ml/src/optimizer/initial_weights_generator/initial_weights_generator.dart';
import 'package:linalg/vector.dart';

class ZeroWeightsGenerator implements InitialWeightsGenerator {
  @override
  SIMDVector generate(int length) => Float32x4VectorFactory.from(List.filled(length, 0.0));
}
