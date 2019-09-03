import 'package:ml_algo/src/solver/non_linear/decision_tree/decision_tree_leaf_label.dart';
import 'package:ml_linalg/matrix.dart';

abstract class DecisionTreeLeafLabelFactory {
  DecisionTreeLeafLabel create(Matrix samples, int targetIdx,
      bool isClassLabelNominal);
}
