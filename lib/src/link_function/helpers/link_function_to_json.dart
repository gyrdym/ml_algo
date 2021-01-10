import 'package:ml_algo/src/link_function/inverse_logit_link_function.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/link_function/link_function_encoded_values.dart';
import 'package:ml_algo/src/link_function/softmax_link_function.dart';

String linkFunctionToJson(LinkFunction linkFunction) {
  if (linkFunction is InverseLogitLinkFunction) {
    return inverseLogitLinkFunctionEncoded;
  }

  if (linkFunction is SoftmaxLinkFunction) {
    return softmaxLinkFunctionEncoded;
  }

  throw UnsupportedError('Unsupported link function type '
      '`${linkFunction.runtimeType}`');
}
