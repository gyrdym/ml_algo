import 'package:ml_algo/src/score_to_prob_link_function/link_function.dart';

abstract class ScoreToProbLinkFunctionFactory {
  ScoreToProbLinkFunction<T> create<T>();
}