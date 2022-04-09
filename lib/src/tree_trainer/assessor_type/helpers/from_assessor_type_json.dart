import 'package:ml_algo/src/tree_trainer/assessor_type/assessor_type.dart';
import 'package:ml_algo/src/tree_trainer/assessor_type/assessor_type_json_keys.dart';

TreeAssessorType fromAssessorTypeJson(String? json) {
  switch (json) {
    case treeAssessorTypeGiniJsonValue:
      return TreeAssessorType.gini;

    case treeAssessorTypeMajorityJsonValue:
    default:
      return TreeAssessorType.majority;
  }
}
