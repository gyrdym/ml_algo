import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor_type_json_keys.dart';

String toTreeAssessorTypeJson(TreeAssessorType type) {
  switch (type) {
    case TreeAssessorType.gini:
      return treeAssessorTypeGiniJsonValue;

    case TreeAssessorType.majority:
      return treeAssessorTypeMajorityJsonValue;
  }
}
