import 'package:dart_ml/src/core/score_function/score_function.dart';
import 'package:simd_vector/vector.dart';

class LinearScore implements ScoreFunction {
  const LinearScore();

  double score(Float32x4Vector w, Float32x4Vector x) => w.dot(x);
}