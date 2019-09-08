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
      [
        this.level = 0,
      ]
  );

  final List<DecisionTreeNode> children;
  final DecisionTreeLeafLabel label;
  final TestSamplePredicate testSample;
  final double splittingNumericalValue;
  final dynamic splittingNominalValue;
  final int splittingIdx;
  final int level;

  bool get isLeaf => children == null || children.isEmpty;

  List<List<DecisionTreeNode>> buildSchema() {
    var levelNodes = <DecisionTreeNode>[];
    final queue = [this];
    final levels = <List<DecisionTreeNode>>[];
    int level = this.level;

    while (queue.isNotEmpty) {
      final node = queue.removeAt(0);
      if (level != node.level) {
        levels.add(levelNodes);
        levelNodes = [];
      }
      level = node.level;
      levelNodes.add(node);
      if (!node.isLeaf) {
        node.children.forEach(queue.add);
      }
    }

    if (levelNodes.isNotEmpty) {
      levels.add(levelNodes);
    }

    return levels;
  }
}
