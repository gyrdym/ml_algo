import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor_type_json_keys.dart';

TreeAssessorType fromTreeAssessorTypeJson(String? json) {
  switch (json) {
    case treeAssessorTypeGiniJsonValue:
      return TreeAssessorType.gini;

    case treeAssessorTypeMajorityJsonValue:
    default:
      return TreeAssessorType.majority;
  }
}
