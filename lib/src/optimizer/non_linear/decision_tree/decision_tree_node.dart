import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_stump.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decisition_tree_base_node.dart';
import 'package:ml_linalg/vector.dart';
import 'package:xrange/zrange.dart';

class DecisionTreeNode extends DecisionTreeBaseNode {
  DecisionTreeNode(double splittingValue, List<Vector> categoricalValues,
      ZRange splittingColumnRange, this.children) :
        super(splittingValue, categoricalValues, splittingColumnRange);

  DecisionTreeNode.leaf() : children = null, super(null, null, null);

  DecisionTreeNode.fromStump(DecisionTreeStump stump, this.children) :
        super(stump.splittingValue, stump.categoricalValues,
          stump.splittingColumnRange);

  final Iterable<DecisionTreeNode> children;

  bool get isLeaf => children == null || children.isEmpty;
}
