import 'package:ml_algo/src/solver/non_linear/decision_tree/decision_tree_leaf_label.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:xrange/zrange.dart';

abstract class DecisionTreeLeafLabelFactory {
  DecisionTreeLeafLabel create(Matrix samples, ZRange outcomesColumnRange,
      bool isClassLabelNominal);
}
