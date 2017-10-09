part of 'package:dart_ml/src/core/implementation.dart';

class _ZeroWeightsGenerator implements InitialWeightsGenerator {
  Float32x4Vector generate(int length) => new Float32x4Vector.from(new List.filled(length, 0.0));
}
