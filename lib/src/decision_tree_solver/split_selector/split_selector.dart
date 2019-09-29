import 'package:ml_algo/src/decision_tree_solver/decision_tree_node.dart';
import 'package:ml_linalg/matrix.dart';

abstract class SplitSelector {
  Map<DecisionTreeNode, Matrix> select(Matrix samples, int targetId,
      Iterable<int> featuresColumnIdxs,
      [Map<int, List<num>> columnIdToUniqueValues]);
}
