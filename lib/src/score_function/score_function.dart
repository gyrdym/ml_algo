library score_function;

import 'package:simd_vector/vector.dart';

part 'linear.dart';

abstract class ScoreFunction {
  double score(Float32x4Vector w, Float32x4Vector x);

  factory ScoreFunction.Linear() => const _LinearScore();
}
