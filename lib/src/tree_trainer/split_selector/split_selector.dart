import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';
import 'package:ml_linalg/matrix.dart';

abstract class TreeSplitSelector {
  Map<TreeNode, Matrix> select(Matrix samples, int targetId,
      Iterable<int> featuresColumnIdxs,
      [Map<int, List<num>> columnIdToUniqueValues]);
}
