import 'dart:typed_data';

import 'package:dart_ml/src/optimizer/initial_weights_generator/initial_weights_generator.dart';
import 'package:ml_linalg/linalg.dart';

class ZeroWeightsGenerator implements InitialWeightsGenerator<Float32x4> {
  @override
  MLVector<Float32x4> generate(int length) =>
      Float32x4VectorFactory.from(List.filled(length, 0.0));
}
