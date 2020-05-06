import 'package:ml_algo/src/common/dtype_serializer/dtype_encoded_values.dart';
import 'package:ml_algo/src/common/dtype_serializer/from_dtype_json.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:test/test.dart';

void main() {
  group('fromDTypeJson', () {
    test('should decode float32 type', () {
      expect(fromDTypeJson(dTypeFloat32EncodedValue), DType.float32);
    });

    test('should decode float64 type', () {
      expect(fromDTypeJson(dTypeFloat64EncodedValue), DType.float64);
    });

    test('should return null in case of unknown json string', () {
      expect(fromDTypeJson('unknown_encoded_value'), isNull);
    });

    test('should return null if null is passed as encoded string', () {
      expect(fromDTypeJson(null), isNull);
    });
  });
}
