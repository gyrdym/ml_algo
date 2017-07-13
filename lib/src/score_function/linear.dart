part of score_function;

class _LinearScore implements ScoreFunction {
  const _LinearScore();

  double score(Float32x4Vector w, Float32x4Vector x) => w.dot(x);
}