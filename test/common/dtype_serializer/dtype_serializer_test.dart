import 'package:ml_algo/src/common/dtype_serializer/dtype_serialized_value.dart';
import 'package:ml_algo/src/common/dtype_serializer/dtype_serializer.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:test/test.dart';

void main() {
  group('DTypeSerializer', () {
    final serializer = const DTypeSerializer();

    test('should serialize string properly', () {
      expect(serializer.serialize(DType.float32), float32Serialized);
      expect(serializer.serialize(DType.float64), float64Serialized);
    });

    test('should throw an unsupported error if unknown value to be '
        'serialized', () {
      expect(() => serializer.serialize(null), throwsUnsupportedError);
    });

    test('should serialize a value properly', () {
      expect(serializer.deserialize(float32Serialized), DType.float32);
      expect(serializer.deserialize(float64Serialized), DType.float64);
    });

    test('should throw an unsupported error if unknown string to be '
        'deserialized', () {
      expect(() => serializer.deserialize('some_unknown_string'),
          throwsUnsupportedError);
    });
  });
}
