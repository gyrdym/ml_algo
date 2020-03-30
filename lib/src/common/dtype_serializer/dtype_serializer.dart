import 'package:ml_algo/src/common/dtype_serializer/dtype_serialized_value.dart';
import 'package:ml_algo/src/common/serializable/primitive_serializer.dart';
import 'package:ml_linalg/dtype.dart';

class DTypeSerializer implements PrimitiveSerializer<DType> {
  const DTypeSerializer();

  @override
  DType deserialize(String serializedValue) {
    switch (serializedValue) {
      case float32Serialized:
        return DType.float32;

      case float64Serialized:
        return DType.float64;

      default:
        throw UnsupportedError('Unsupported serialized value: $serializedValue');
    }
  }

  @override
  String serialize(DType value) {
    switch (value) {
      case DType.float32:
        return float32Serialized;

      case DType.float64:
        return float32Serialized;
    };

    throw UnsupportedError('Unsupported dtype value: $value');
  }
}
