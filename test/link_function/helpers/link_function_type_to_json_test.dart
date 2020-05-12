import 'package:ml_algo/src/link_function/helpers/link_function_type_to_json.dart';
import 'package:ml_algo/src/link_function/link_function_encoded_types.dart';
import 'package:ml_algo/src/link_function/link_function_type.dart';
import 'package:test/test.dart';

void main() {
  group('linkFunctionTypeToJson', () {
    test('should encode softmax link function type', () {
      final encoded = linkFunctionTypeToJson(LinkFunctionType.softmax);
      expect(encoded, softmaxLinkFunctionEncodedType);
    });

    test('should encode inverse logit link function type', () {
      final encoded = linkFunctionTypeToJson(LinkFunctionType.inverseLogit);
      expect(encoded, inverseLogitLinkFunctionEncodedType);
    });

    test('should throw an error if null is passed as a link function type', () {
      final actual = () => linkFunctionTypeToJson(null);
      expect(actual, throwsUnsupportedError);
    });
  });
}
