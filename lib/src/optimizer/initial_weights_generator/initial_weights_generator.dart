import 'package:simd_vector/vector.dart';

abstract class InitialWeightsGenerator {
  Float32x4Vector generate(int length);
}
