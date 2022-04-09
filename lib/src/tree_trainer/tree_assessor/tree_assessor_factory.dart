import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor_type.dart';

abstract class TreeAssessorFactory {
  TreeAssessor createByType(TreeAssessorType type);
}
