import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';
import 'package:ml_linalg/matrix.dart';

abstract class TreeSplitter {
  Map<TreeNode, Matrix> split(Matrix samples, int splittingIdx, int targetId,
      [List<num>? uniqueValues]);
}
