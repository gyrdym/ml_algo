import 'package:ml_algo/src/tree_trainer/tree_node/decision_tree_node/decision_tree_node.dart';
import 'package:ml_linalg/matrix.dart';

abstract class TreeTrainer {
  DecisionTreeNode train(Matrix samples);
}
