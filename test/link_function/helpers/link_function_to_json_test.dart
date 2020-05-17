import 'package:ml_algo/src/link_function/helpers/link_function_to_json.dart';
import 'package:ml_algo/src/link_function/link_function_encoded_values.dart';
import 'package:ml_algo/src/link_function/logit/float32_inverse_logit_function.dart';
import 'package:ml_algo/src/link_function/softmax/float32_softmax_link_function.dart';
import 'package:test/test.dart';

void main() {
  group('linkFunctionToJson', () {
    test('should encode float32 inverse logit', () {
      final float32InverseLogit = const Float32InverseLogitLinkFunction();
      final encoded = linkFunctionToJson(float32InverseLogit);

      expect(encoded, float32InverseLogitLinkFunctionEncoded);
    });

    test('should encode float32 softmax link function', () {
      final float32Softmax = const Float32SoftmaxLinkFunction();
      final encoded = linkFunctionToJson(float32Softmax);

      expect(encoded, float32SoftmaxLinkFunctionEncoded);
    });

    test('should throw an error if null is passed as the argument', () {
      final actual = () => linkFunctionToJson(null);

      expect(actual, throwsUnsupportedError);
    });
  });
}
