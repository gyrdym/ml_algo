import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_stump.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:xrange/zrange.dart';

class DecisionTreeNode {
  DecisionTreeNode(this.splittingValue, this.categoricalValues,
      this.splittingColumnRange, this.children, this.observations);

  DecisionTreeNode.leaf(this.observations) :
      splittingValue = null,
      categoricalValues = null,
      splittingColumnRange = null,
      children = null;

  DecisionTreeNode.fromStump(DecisionTreeStump stump, this.children) :
      splittingValue = stump.splittingValue,
      categoricalValues = stump.categoricalValues,
      splittingColumnRange = stump.splittingColumnRange,
      observations = null;

  final double splittingValue;
  final List<Vector> categoricalValues;
  final ZRange splittingColumnRange;
  final Iterable<DecisionTreeNode> children;
  final Matrix observations;

  bool get isLeaf => observations != null;
}
