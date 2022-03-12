import 'package:ml_algo/src/tree_trainer/tree_node/decision_tree_node/tree_node_json_keys.dart';
import 'package:test/test.dart';

void main() {
  group('Tree node json keys', () {
    test('should contain json key for children field', () {
      expect(childrenJsonKey, 'CN');
    });

    test('should contain json key for label field', () {
      expect(labelJsonKey, 'LB');
    });

    test('should contain json key for splitting predicate type field', () {
      expect(predicateTypeJsonKey, 'PT');
    });

    test('should contain json key for splitting value field', () {
      expect(splittingValueJsonKey, 'SV');
    });

    test('should contain json key for splitting index field', () {
      expect(splittingIndexJsonKey, 'SI');
    });

    test('should contain json key for level field', () {
      expect(levelJsonKey, 'LV');
    });
  });
}
