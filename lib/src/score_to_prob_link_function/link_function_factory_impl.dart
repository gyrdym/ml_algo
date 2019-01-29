import 'dart:typed_data';

import 'package:ml_algo/src/score_to_prob_link_function/float32x4_logit_link_function_factory.dart';
import 'package:ml_algo/src/score_to_prob_link_function/link_function_factory.dart';

class ScoreToProbLinkFunctionFactoryImpl implements ScoreToProbLinkFunctionFactory {
  const ScoreToProbLinkFunctionFactoryImpl();

  @override
  Function fromDataType(Type type) {
    switch (type) {
      case Float32x4:
        return float32x4LogitLinkFunctionFactory();
      default:
        throw UnimplementedError();
    }
  }
}
