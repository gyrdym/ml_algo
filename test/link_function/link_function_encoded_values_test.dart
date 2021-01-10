import 'package:ml_algo/src/link_function/link_function_encoded_values.dart';
import 'package:test/test.dart';

void main() {
  group('Link function encoded values (obsolete)', () {
    test('should contain a proper encoded value for float32 inverse logit', () {
      expect(v1_float32InverseLogitLinkFunctionEncoded, 'F32IL');
    });

    test('should contain a proper encoded value for float64 inverse logit', () {
      expect(v1_float64InverseLogitLinkFunctionEncoded, 'F64IL');
    });

    test('should contain a proper encoded value for float32 softmax link '
        'function', () {
      expect(v1_float32SoftmaxLinkFunctionEncoded, 'F32SM');
    });

    test('should contain a proper encoded value for float64 softmax link '
        'function', () {
      expect(v1_float64SoftmaxLinkFunctionEncoded, 'F64SM');
    });
  });
}
