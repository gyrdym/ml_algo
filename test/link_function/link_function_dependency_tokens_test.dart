import 'package:ml_algo/src/link_function/link_function_dependency_tokens.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:test/test.dart';

void main() {
  group('Link function dependency tokens', () {
    test('should look up a correct token by DType for float32 inverse logit', () {
      expect(dTypeToInverseLogitLinkFunctionToken[DType.float32],
          float32InverseLogitLinkFunctionToken);
    });

    test('should look up a correct token by DType for float64 inverse logit', () {
      expect(dTypeToInverseLogitLinkFunctionToken[DType.float64],
          float64InverseLogitLinkFunctionToken);
    });

    test('should look up a correct token by DType for float32 softmax link '
        'function', () {
      expect(dTypeToSoftmaxLinkFunctionToken[DType.float32],
          float32SoftmaxLinkFunctionToken);
    });

    test('should look up a correct token by DType for float64 softmax link '
        'function', () {
      expect(dTypeToSoftmaxLinkFunctionToken[DType.float64],
          float64SoftmaxLinkFunctionToken);
    });
  });
}
