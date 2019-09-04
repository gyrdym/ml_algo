import 'package:ml_algo/src/solver/non_linear/decision_tree/decision_tree_node.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:ml_linalg/matrix.dart';

abstract class NominalSplitter {
  Map<DecisionTreeNode, Matrix> split(Matrix samples, int splittingIdx,
      List<dynamic> uniqueValues);
}
