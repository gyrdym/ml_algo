import 'package:ml_algo/src/link_function/helpers/link_function_to_json.dart';
import 'package:ml_algo/src/link_function/inverse_logit_link_function.dart';
import 'package:ml_algo/src/link_function/link_function_encoded_values.dart';
import 'package:ml_algo/src/link_function/softmax_link_function.dart';
import 'package:test/test.dart';

void main() {
  group('linkFunctionToJson', () {
    test('should encode inverse logit', () {
      final inverseLogit = const InverseLogitLinkFunction();
      final encoded = linkFunctionToJson(inverseLogit);

      expect(encoded, inverseLogitLinkFunctionEncoded);
    });

    test('should encode softmax link function', () {
      final softmaxFunction = const SoftmaxLinkFunction();
      final encoded = linkFunctionToJson(softmaxFunction);

      expect(encoded, softmaxLinkFunctionEncoded);
    });
  });
}
