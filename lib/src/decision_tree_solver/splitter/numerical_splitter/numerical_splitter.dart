import 'package:ml_algo/src/decision_tree_solver/decision_tree_node.dart';
import 'package:ml_linalg/matrix.dart';

abstract class NumericalSplitter {
  Map<DecisionTreeNode, Matrix> split(Matrix samples, int splittingIdx,
      double splittingValue);
}
