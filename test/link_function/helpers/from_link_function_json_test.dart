import 'package:ml_algo/src/link_function/helpers/from_link_function_json.dart';
import 'package:ml_algo/src/link_function/inverse_logit_link_function.dart';
import 'package:ml_algo/src/link_function/link_function_encoded_values.dart';
import 'package:ml_algo/src/link_function/softmax_link_function.dart';
import 'package:test/test.dart';

void main() {
  group('fromLinkFunctionJson', () {
    test(
        'should decode inverse logit link function, '
        'v1_float32InverseLogitLinkFunctionEncoded', () {
      final logitFunction = const InverseLogitLinkFunction();
      final decoded =
          fromLinkFunctionJson(v1_float32InverseLogitLinkFunctionEncoded);
      expect(decoded, same(logitFunction));
    });

    test(
        'should decode inverse logit link function, '
        'v1_float64InverseLogitLinkFunctionEncoded', () {
      final logitFunction = const InverseLogitLinkFunction();
      final decoded =
          fromLinkFunctionJson(v1_float64InverseLogitLinkFunctionEncoded);
      expect(decoded, same(logitFunction));
    });

    test(
        'should decode inverse logit link function, '
        'inverseLogitLinkFunctionEncoded', () {
      final logitFunction = const InverseLogitLinkFunction();
      final decoded = fromLinkFunctionJson(inverseLogitLinkFunctionEncoded);
      expect(decoded, same(logitFunction));
    });

    test(
        'should decode softmax link function, '
        'v1_float32SoftmaxLinkFunctionEncoded', () {
      final softmaxFunction = const SoftmaxLinkFunction();
      final decoded =
          fromLinkFunctionJson(v1_float32SoftmaxLinkFunctionEncoded);
      expect(decoded, same(softmaxFunction));
    });

    test(
        'should decode softmax link function, '
        'v1_float64SoftmaxLinkFunctionEncoded', () {
      final softmaxFunction = const SoftmaxLinkFunction();
      final decoded =
          fromLinkFunctionJson(v1_float64SoftmaxLinkFunctionEncoded);
      expect(decoded, same(softmaxFunction));
    });

    test(
        'should decode softmax link function, '
        'softmaxLinkFunctionEncoded', () {
      final softmaxFunction = const SoftmaxLinkFunction();
      final decoded = fromLinkFunctionJson(softmaxLinkFunctionEncoded);
      expect(decoded, same(softmaxFunction));
    });

    test('should throw an error if unknow string is passed', () {
      final actual = () => fromLinkFunctionJson('unknown_string');
      expect(actual, throwsUnsupportedError);
    });
  });
}
