import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_base_node.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_leaf_label.dart';
import 'package:ml_linalg/vector.dart';
import 'package:xrange/zrange.dart';

class DecisionTreeNode extends DecisionTreeBaseNode {
  DecisionTreeNode(
      FilterPredicate isSampleAcceptable,
      double splittingNumericalValue,
      Vector splittingNominalValue,
      ZRange splittingColumnRange,
      this.children,
      this.label,
  ) : super(
      isSampleAcceptable,
      splittingNumericalValue,
      splittingNominalValue,
      splittingColumnRange);

  DecisionTreeNode.fromOther(
      DecisionTreeBaseNode stump,
      this.children,
      this.label) :
        super(
          stump.isSampleAcceptable,
          stump.splittingNumericalValue,
          stump.splittingNominalValue,
          stump.splittingColumnRange);

  final List<DecisionTreeNode> children;
  final DecisionTreeLeafLabel label;

  bool get isLeaf => children == null || children.isEmpty;
}
