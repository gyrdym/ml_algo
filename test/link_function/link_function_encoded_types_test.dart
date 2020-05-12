import 'package:ml_algo/src/link_function/link_function_encoded_types.dart';
import 'package:test/test.dart';

void main() {
  group('Link function encoded types', () {
    test('should contain encoded value for inverse logit link function type', () {
      expect(inverseLogitLinkFunctionEncodedType, 'IL');
    });

    test('should contain encoded value for softmax link function type', () {
      expect(softmaxLinkFunctionEncodedType, 'S');
    });
  });
}
