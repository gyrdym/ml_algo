import 'package:simd_vector/vector.dart';

abstract class ScoreFunction {
  double score(Float32x4Vector w, Float32x4Vector x);
}
