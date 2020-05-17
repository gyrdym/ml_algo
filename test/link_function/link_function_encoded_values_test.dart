import 'package:ml_algo/src/link_function/link_function_encoded_values.dart';
import 'package:test/test.dart';

void main() {
  group('Link function encoded values', () {
    test('should contain a proper encoded value for float32 inverse logit', () {
      expect(float32InverseLogitLinkFunctionEncoded, 'F32IL');
    });

    test('should contain a proper encoded value for float32 softmax link '
        'function', () {
      expect(float32SoftmaxLinkFunctionEncoded, 'F32SM');
    });
  });
}
