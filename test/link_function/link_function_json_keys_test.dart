import 'package:ml_algo/src/link_function/link_function_json_keys.dart';
import 'package:test/test.dart';

void main() {
  group('Link function json keys', () {
    test('should contain a json key for type field', () {
      expect(linkFunctionTypeJsonKey, 'LT');
    });

    test('should contain a json key for dtype field', () {
      expect(linkFunctionDTypeJsonKey, 'DT');
    });
  });
}
