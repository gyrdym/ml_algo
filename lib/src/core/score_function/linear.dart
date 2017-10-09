part of 'package:dart_ml/src/core/implementation.dart';

class _LinearScore implements ScoreFunction {
  const _LinearScore();

  double score(Float32x4Vector w, Float32x4Vector x) => w.dot(x);
}