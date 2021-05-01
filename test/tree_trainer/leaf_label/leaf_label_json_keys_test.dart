import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label_json_keys.dart';
import 'package:test/test.dart';

void main() {
  group('Leaf label json keys', () {
    test('should have a key for value field', () {
      expect(leafLabelValueJsonKey, 'V');
    });

    test('should have a key for probability field', () {
      expect(leafLabelProbabilityJsonKey, 'P');
    });
  });
}
