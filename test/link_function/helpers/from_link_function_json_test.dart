import 'package:ml_algo/src/link_function/helpers/from_link_function_json.dart';
import 'package:ml_algo/src/link_function/link_function_encoded_values.dart';
import 'package:ml_algo/src/link_function/logit/float32_inverse_logit_function.dart';
import 'package:ml_algo/src/link_function/softmax/float32_softmax_link_function.dart';
import 'package:test/test.dart';

void main() {
  group('fromLinkFunctionJson', () {
    test('should decode Float32 based inverse logit link function', () {
      final logitFunction = const Float32InverseLogitLinkFunction();
      final decoded = fromLinkFunctionJson(
          float32InverseLogitLinkFunctionEncoded);
      expect(decoded, same(logitFunction));
    });

    test('should decode Float32 based softmax link function', () {
      final logitFunction = const Float32SoftmaxLinkFunction();
      final decoded = fromLinkFunctionJson(float32SoftmaxLinkFunctionEncoded);
      expect(decoded, same(logitFunction));
    });

    test('should throw an error if unknow string is passed', () {
      final actual = () => fromLinkFunctionJson('unknown_string');
      expect(actual, throwsUnsupportedError);
    });
  });
}
