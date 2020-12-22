import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_json_keys.dart';
import 'package:test/test.dart';

void main() {
  group('Decision tree json keys', () {
    test('should have a proper key for dType field', () {
      expect(decisionTreeClassifierDTypeJsonKey, 'DT');
    });

    test('should have a proper key for target column name field', () {
      expect(decisionTreeClassifierTargetColumnNameJsonKey, 'T');
    });

    test('should have a proper key for tree root node field', () {
      expect(decisionTreeClassifierTreeRootNodeJsonKey, 'R');
    });
  });
}
