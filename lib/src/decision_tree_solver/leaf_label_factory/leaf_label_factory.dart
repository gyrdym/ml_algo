import 'package:ml_algo/src/decision_tree_solver/decision_tree_leaf_label.dart';
import 'package:ml_linalg/matrix.dart';

abstract class DecisionTreeLeafLabelFactory {
  DecisionTreeLeafLabel create(Matrix samples, int targetIdx);
}
