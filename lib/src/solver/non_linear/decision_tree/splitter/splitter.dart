import 'package:ml_algo/src/solver/non_linear/decision_tree/decision_tree_node.dart';
import 'package:ml_linalg/matrix.dart';

abstract class Splitter {
  Map<DecisionTreeNode, Matrix> split(Matrix samples, int splittingIdx,
      int targetId, [List<num> uniqueValues]);
}
