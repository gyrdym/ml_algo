import 'package:simd_vector/vector.dart';

abstract class InitialWeightsGenerator {
  Vector generate(int length);
}
