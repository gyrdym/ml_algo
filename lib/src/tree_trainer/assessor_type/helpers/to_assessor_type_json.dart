import 'package:ml_algo/src/tree_trainer/assessor_type/assessor_type.dart';
import 'package:ml_algo/src/tree_trainer/assessor_type/assessor_type_json_keys.dart';

String toAssessorTypeJson(TreeAssessorType type) {
  switch (type) {
    case TreeAssessorType.gini:
      return treeAssessorTypeGiniJsonValue;

    case TreeAssessorType.majority:
      return treeAssessorTypeMajorityJsonValue;
  }
}
