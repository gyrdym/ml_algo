import 'package:simd_vector/vector.dart';

abstract class Classifier {
  Float32x4Vector predictClasses(List<Float32x4Vector> features);
}