import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';

class _LeveledNode {
  _LeveledNode(this.node, this.level);

  final TreeNode node;
  final int level;
}

TreeNode _createFakeNode() {
  return TreeNode(null, null, null, null, null);
}

List<List<TreeNode>> getTreeLevels(TreeNode node, int depth) {
  var currentLevel = -1;
  final queue = [_LeveledNode(node, 0)];
  final levels = <List<TreeNode>>[];

  while (queue.isNotEmpty) {
    final node = queue.removeAt(0);
    final children = node.node.children?.isNotEmpty == true
        ? node.node.children!
        : node.level + 1 == depth
            ? <TreeNode>[]
            : [_createFakeNode(), _createFakeNode()];

    if (currentLevel != node.level) {
      levels.add([node.node]);
    } else {
      levels[currentLevel].add(node.node);
    }

    currentLevel = node.level;

    queue.addAll(children.map((child) => _LeveledNode(child, node.level + 1)));
  }

  return levels;
}
