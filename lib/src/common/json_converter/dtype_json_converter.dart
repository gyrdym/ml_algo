import 'package:json_annotation/json_annotation.dart';
import 'package:ml_algo/src/common/constants/default_parameters/common.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/dtype_to_json.dart';
import 'package:ml_linalg/from_dtype_json.dart';

class DTypeJsonConverter implements JsonConverter<DType, String> {
  const DTypeJsonConverter();

  @override
  DType fromJson(String? json) => fromDTypeJson(json) ?? dTypeDefaultValue;

  @override
  String toJson(DType dtype) => dTypeToJson(dtype)!;
}
