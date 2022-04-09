import 'package:ml_algo/src/tree_trainer/split_assessor/split_assessor.dart';
import 'package:ml_algo/src/tree_trainer/assessor_type/assessor_type.dart';

abstract class TreeSplitAssessorFactory {
  TreeSplitAssessor createByType(TreeAssessorType type);
}
