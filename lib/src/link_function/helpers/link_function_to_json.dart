import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/link_function/link_function_encoded_values.dart';
import 'package:ml_algo/src/link_function/logit/float32_inverse_logit_function.dart';
import 'package:ml_algo/src/link_function/softmax/float32_softmax_link_function.dart';

String linkFunctionToJson(LinkFunction linkFunction) {
  if (linkFunction is Float32InverseLogitLinkFunction) {
    return float32InverseLogitLinkFunctionEncoded;
  }

  if (linkFunction is Float32SoftmaxLinkFunction) {
    return float32SoftmaxLinkFunctionEncoded;
  }

  throw UnsupportedError('Unsupported link function type '
      '`${linkFunction.runtimeType}`');
}
