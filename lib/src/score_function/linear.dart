part of score_function;

class _LinearScore implements ScoreFunction {
  const _LinearScore();

  double score(Vector w, Vector x) => w.dot(x);
}