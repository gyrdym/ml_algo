import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/link_function/link_function_encoded_values.dart';
import 'package:ml_algo/src/link_function/logit/float32_inverse_logit_function.dart';
import 'package:ml_algo/src/link_function/logit/float64_inverse_logit_function.dart';
import 'package:ml_algo/src/link_function/softmax/float32_softmax_link_function.dart';
import 'package:ml_algo/src/link_function/softmax/float64_softmax_link_function.dart';

LinkFunction fromLinkFunctionJson(String encodedLinkFunction) {
  switch (encodedLinkFunction) {
    case float32InverseLogitLinkFunctionEncoded:
      return const Float32InverseLogitLinkFunction();

    case float64InverseLogitLinkFunctionEncoded:
      return const Float64InverseLogitLinkFunction();

    case float32SoftmaxLinkFunctionEncoded:
      return const Float32SoftmaxLinkFunction();

    case float64SoftmaxLinkFunctionEncoded:
      return const Float64SoftmaxLinkFunction();
  }

  throw UnsupportedError('Unsupported encoded link function '
      '`$encodedLinkFunction`');
}
