import 'dart:typed_data';

import 'package:ml_algo/src/score_to_prob_link_function/float32x4_link_function.dart';
import 'package:ml_algo/src/score_to_prob_link_function/link_function.dart';
import 'package:ml_algo/src/score_to_prob_link_function/link_function_factory.dart';

class ScoreToProbLinkFunctionFactoryImpl implements ScoreToProbLinkFunctionFactory {
  const ScoreToProbLinkFunctionFactoryImpl();

  @override
  ScoreToProbLinkFunction<T> create<T>() {
    switch (T) {
      case Float32x4:
        return vectorizedLogitLink as ScoreToProbLinkFunction<T>;
      default:
        throw UnimplementedError();
    }
  }
}
