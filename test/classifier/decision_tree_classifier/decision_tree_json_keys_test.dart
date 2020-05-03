import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_json_keys.dart';
import 'package:test/test.dart';

void main() {
  group('Decision tree json keys', () {
    test('should have a proper key for dType field', () {
      expect(dTypeJsonKey, 'DT');
    });

    test('should have a proper key for target column name field', () {
      expect(targetColumnNameJsonKey, 'T');
    });

    test('should have a proper key for tree root node field', () {
      expect(treeRootNodeJsonKey, 'R');
    });
  });
}
