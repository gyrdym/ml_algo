import 'package:ml_algo/src/tree_trainer/tree_assessor/helpers/from_tree_assessor_type_json.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor_type_json_keys.dart';
import 'package:test/test.dart';

void main() {
  group('fromTreeAssessorTypeJson', () {
    test('should convert json representation of gini assessor type', () {
      expect(fromTreeAssessorTypeJson(treeAssessorTypeGiniJsonValue),
          TreeAssessorType.gini);
    });

    test('should convert json representation of majority assessor type', () {
      expect(fromTreeAssessorTypeJson(treeAssessorTypeMajorityJsonValue),
          TreeAssessorType.majority);
    });
  });
}
