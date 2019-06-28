import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_leaf_label.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_stump.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decisition_tree_base_node.dart';
import 'package:ml_linalg/vector.dart';
import 'package:xrange/zrange.dart';

class DecisionTreeNode extends DecisionTreeBaseNode {
  DecisionTreeNode(double splittingValue, List<Vector> categoricalValues,
      ZRange splittingColumnRange, this.children) :
        label = null,
        super(splittingValue, categoricalValues, splittingColumnRange);

  DecisionTreeNode.leaf(this.label) : children = null, super(null, null, null);

  DecisionTreeNode.fromStump(DecisionTreeStump stump, this.children) :
        label = null,
        super(stump.splittingValue, stump.categoricalValues,
          stump.splittingColumnRange);

  final Iterable<DecisionTreeNode> children;
  final DecisionTreeLeafLabel label;

  bool get isLeaf => children == null || children.isEmpty;
}
