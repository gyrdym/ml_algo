import 'package:ml_algo/src/common/dtype_serializer/dtype_encoded_values.dart';
import 'package:ml_linalg/dtype.dart';

DType fromDTypeJson(String json) {
  switch (json) {
    case dTypeFloat32EncodedValue:
      return DType.float32;

    case dTypeFloat64EncodedValue:
      return DType.float64;

    default:
      return null;
  }
}
