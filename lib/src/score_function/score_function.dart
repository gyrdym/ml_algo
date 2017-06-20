library score_function;

import 'package:dart_ml/src/math/vector/vector.dart';

part 'linear.dart';

abstract class ScoreFunction {
  double score(Vector w, Vector x);

  factory ScoreFunction.Linear() => const _LinearScore();
}
