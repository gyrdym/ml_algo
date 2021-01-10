import 'package:ml_algo/src/link_function/inverse_logit_link_function.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/link_function/link_function_encoded_values.dart';
import 'package:ml_algo/src/link_function/softmax_link_function.dart';

LinkFunction fromLinkFunctionJson(String encodedLinkFunction) {
  switch (encodedLinkFunction) {
    case v1_float32InverseLogitLinkFunctionEncoded:
    case v1_float64InverseLogitLinkFunctionEncoded:
    case inverseLogitLinkFunctionEncoded:
      return const InverseLogitLinkFunction();

    case v1_float32SoftmaxLinkFunctionEncoded:
    case v1_float64SoftmaxLinkFunctionEncoded:
    case softmaxLinkFunctionEncoded:
      return const SoftmaxLinkFunction();
  }

  throw UnsupportedError('Unsupported encoded link function '
      '`$encodedLinkFunction`');
}
