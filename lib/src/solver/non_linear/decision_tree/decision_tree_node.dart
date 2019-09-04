import 'package:ml_algo/src/solver/non_linear/decision_tree/decision_tree_leaf_label.dart';
import 'package:ml_linalg/vector.dart';

typedef TestSamplePredicate = bool Function(Vector sample);

class DecisionTreeNode {
  DecisionTreeNode(
      this.testSample,
      this.splittingNumericalValue,
      this.splittingNominalValue,
      this.splittingIdx,
      this.children,
      this.label,
  );

  final List<DecisionTreeNode> children;
  final DecisionTreeLeafLabel label;
  final TestSamplePredicate testSample;
  final double splittingNumericalValue;
  final dynamic splittingNominalValue;
  final int splittingIdx;

  bool get isLeaf => children == null || children.isEmpty;
}
