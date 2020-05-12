import 'package:ml_algo/src/link_function/helpers/from_link_function_type_json.dart';
import 'package:ml_algo/src/link_function/link_function_encoded_types.dart';
import 'package:ml_algo/src/link_function/link_function_type.dart';
import 'package:test/test.dart';

void main() {
  group('fromLinkFunctionTypeJson', () {
    test('should decode inverse logit type', () {
      final decoded = fromLinkFunctionTypeJson(
          inverseLogitLinkFunctionEncodedType);
      expect(decoded, LinkFunctionType.inverseLogit);
    });

    test('should decode softmax type', () {
      final decoded = fromLinkFunctionTypeJson(
          softmaxLinkFunctionEncodedType);
      expect(decoded, LinkFunctionType.softmax);
    });

    test('should throw an error if unknown encoded value is passed', () {
      final actual = () => fromLinkFunctionTypeJson('some_unknown_value');
      expect(actual, throwsUnsupportedError);
    });
  });
}
