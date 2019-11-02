import 'package:ml_algo/src/tree_solver/split_assessor/split_assessor.dart';
import 'package:ml_algo/src/tree_solver/split_assessor/split_assessor_type.dart';

abstract class TreeSplitAssessorFactory {
  TreeSplitAssessor createByType(TreeSplitAssessorType type);
}
