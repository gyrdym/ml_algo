import 'package:ml_algo/src/common/dtype_serializer/dtype_encoded_values.dart';
import 'package:test/test.dart';

void main() {
  group('Dtype encoded values', () {
    test('should have a proper value for float32 dtype', () {
      expect(dTypeFloat32EncodedValue, 'F32');
    });

    test('should have a proper value for float64 dtype', () {
      expect(dTypeFloat64EncodedValue, 'F64');
    });
  });
}
