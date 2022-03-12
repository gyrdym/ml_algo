import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';
import 'package:ml_linalg/matrix.dart';

abstract class NumericalTreeSplitter {
  Map<T, Matrix> split<T extends TreeNode>(
      Matrix samples, int splittingIdx, double splittingValue);
}
