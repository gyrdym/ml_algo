import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/link_function/link_function_encoded_values.dart';
import 'package:ml_algo/src/link_function/logit/float32_inverse_logit_function.dart';
import 'package:ml_algo/src/link_function/softmax/float32_softmax_link_function.dart';

LinkFunction fromLinkFunctionJson(String encodedLinkFunction) {
  switch (encodedLinkFunction) {
    case float32InverseLogitLinkFunctionEncoded:
      return const Float32InverseLogitLinkFunction();

    case float32SoftmaxLinkFunctionEncoded:
      return const Float32SoftmaxLinkFunction();
  }

  throw UnsupportedError('Unsupported encoded link function '
      '`$encodedLinkFunction`');
}
