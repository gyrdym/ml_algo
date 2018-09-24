import 'dart:typed_data';

import 'package:dart_ml/src/optimizer/initial_weights_generator/initial_weights_generator.dart';
import 'package:linalg/vector.dart';

class ZeroWeightsGenerator implements InitialWeightsGenerator<Float32x4List, Float32List, Float32x4> {
  @override
  SIMDVector<Float32x4List, Float32List, Float32x4> generate(int length) =>
      Float32x4VectorFactory.from(List.filled(length, 0.0));
}
