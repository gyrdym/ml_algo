import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';

Iterable<TreeNode> fromTreeNodesJson(
    Iterable<Map<String, dynamic>> collection) =>
    collection.map((nodeJson) => TreeNode.fromJson(nodeJson));
