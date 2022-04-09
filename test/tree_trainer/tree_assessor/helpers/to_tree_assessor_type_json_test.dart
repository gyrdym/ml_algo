import 'package:ml_algo/src/tree_trainer/tree_assessor/helpers/to_tree_assessor_type_json.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor_type_json_keys.dart';
import 'package:test/test.dart';

void main() {
  group('toTreeAssessorTypeJson', () {
    test('should convert gini assessor type to json representation', () {
      expect(toTreeAssessorTypeJson(TreeAssessorType.gini),
          treeAssessorTypeGiniJsonValue);
    });

    test('should convert majority assessor type to json representation', () {
      expect(toTreeAssessorTypeJson(TreeAssessorType.majority),
          treeAssessorTypeMajorityJsonValue);
    });
  });
}
