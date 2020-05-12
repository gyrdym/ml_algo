import 'package:ml_algo/src/link_function/link_function_encoded_types.dart';
import 'package:ml_algo/src/link_function/link_function_type.dart';

String linkFunctionTypeToJson(LinkFunctionType type) {
  switch (type) {
    case LinkFunctionType.softmax:
      return softmaxLinkFunctionEncodedType;

    case LinkFunctionType.inverseLogit:
      return inverseLogitLinkFunctionEncodedType;

    default:
      throw UnsupportedError('Unsupported link function type `$type`');
  }
}
