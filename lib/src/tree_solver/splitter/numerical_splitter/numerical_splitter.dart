import 'package:ml_algo/src/tree_solver/tree_node.dart';
import 'package:ml_linalg/matrix.dart';

abstract class NumericalTreeSplitter {
  Map<TreeNode, Matrix> split(Matrix samples, int splittingIdx,
      double splittingValue);
}
