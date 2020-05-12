import 'package:json_annotation/json_annotation.dart';
import 'package:ml_algo/src/common/dtype_serializer/dtype_to_json.dart';
import 'package:ml_algo/src/common/dtype_serializer/from_dtype_json.dart';
import 'package:ml_algo/src/link_function/helpers/from_link_function_type_json.dart';
import 'package:ml_algo/src/link_function/helpers/link_function_type_to_json.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/link_function/link_function_json_keys.dart';
import 'package:ml_algo/src/link_function/link_function_type.dart';
import 'package:ml_algo/src/link_function/logit/float32_inverse_logit_link_function_mixin.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';

part 'inverse_logit_link_function.g.dart';

@JsonSerializable()
class InverseLogitLinkFunction with Float32InverseLogitLinkFunction
    implements LinkFunction {

  InverseLogitLinkFunction(this.dtype);

  @override
  @JsonKey(
    name: linkFunctionTypeJsonKey,
    toJson: linkFunctionTypeToJson,
    fromJson: fromLinkFunctionTypeJson,
  )
  final LinkFunctionType type = LinkFunctionType.inverseLogit;

  @JsonKey(
    name: linkFunctionDTypeJsonKey,
    toJson: dTypeToJson,
    fromJson: fromDTypeJson,
  )
  final DType dtype;

  @override
  Matrix link(Matrix scores) {
    switch (dtype) {
      case DType.float32:
        return getFloat32x4Probabilities(scores);
      default:
        throw UnsupportedError('Unsupported data type - $dtype');
    }
  }
}
