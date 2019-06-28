import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_leaf_label.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:xrange/zrange.dart';

abstract class DecisionTreeLeafLabelFactory {
  DecisionTreeLeafLabel create(Matrix observations, ZRange outcomesColumnRange,
      bool isClassLabelCategorical);
}
