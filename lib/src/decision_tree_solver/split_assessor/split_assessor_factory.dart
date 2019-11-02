import 'package:ml_algo/src/decision_tree_solver/split_assessor/split_assessor.dart';
import 'package:ml_algo/src/decision_tree_solver/split_assessor/split_assessor_type.dart';

abstract class DecisionTreeSplitAssessorFactory {
  DecisionTreeSplitAssessor createByType(DecisionTreeSplitAssessorType type);
}
