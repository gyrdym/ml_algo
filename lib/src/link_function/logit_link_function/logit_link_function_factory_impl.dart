import 'dart:typed_data';

import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/link_function/logit_link_function/float32x4_logit_link_function_factory.dart';
import 'package:ml_algo/src/link_function/logit_link_function/logit_link_function_factory.dart';

class LogitLinkFunctionFactoryImpl implements LogitLinkFunctionFactory {
  @override
  LinkFunction fromDataType(Type dtype) {
    switch (dtype) {
      case Float32x4:
        return float32x4LogitLinkFunctionFactory() as LinkFunction;
      default:
        throw UnimplementedError();
    }
  }
}
