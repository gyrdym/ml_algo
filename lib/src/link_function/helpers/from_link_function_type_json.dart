import 'package:ml_algo/src/link_function/link_function_encoded_types.dart';
import 'package:ml_algo/src/link_function/link_function_type.dart';

LinkFunctionType fromLinkFunctionTypeJson(String encodedType) {
  switch (encodedType) {
    case inverseLogitLinkFunctionEncodedType:
      return LinkFunctionType.inverseLogit;

    case softmaxLinkFunctionEncodedType:
      return LinkFunctionType.softmax;

    default:
      throw UnsupportedError('Unsupported encoded link function type '
          '`$encodedType`');
  }
}
