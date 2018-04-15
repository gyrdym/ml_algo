import 'package:dart_ml/src/core/score_function/linear.dart';
import 'package:dart_ml/src/core/score_function/score_function.dart';

class ScoreFunctionFactory {
  static ScoreFunction Linear() => const LinearScore();
}