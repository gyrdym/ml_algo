import 'package:ml_algo/src/common/dtype_serializer/dtype_encoded_values.dart';
import 'package:ml_algo/src/common/dtype_serializer/dtype_to_json.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:test/test.dart';

void main() {
  group('dTypeToJson', () {
    test('should encode float32 dtype', () {
      expect(dTypeToJson(DType.float32), dTypeFloat32EncodedValue);
    });

    test('should encode float64 dtype', () {
      expect(dTypeToJson(DType.float64), dTypeFloat64EncodedValue);
    });

    test('should return null if dtype is unknown', () {
      expect(dTypeToJson(null), isNull);
    });
  });
}
