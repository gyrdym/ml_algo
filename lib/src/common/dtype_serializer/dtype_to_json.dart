import 'package:ml_algo/src/common/dtype_serializer/dtype_encoded_values.dart';
import 'package:ml_linalg/dtype.dart';

String dTypeToJson(DType dtype) {
  switch (dtype) {
    case DType.float32:
      return dTypeFloat32EncodedValue;

    case DType.float64:
      return dTypeFloat64EncodedValue;

    default:
      return null;
  }
}
