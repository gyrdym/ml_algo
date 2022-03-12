import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:ml_linalg/matrix.dart';

abstract class NominalTreeSplitter {
  Map<T, Matrix> split<T extends TreeNode>(
      Matrix samples, int splittingIdx, List<num> uniqueValues);
}
