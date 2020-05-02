import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';

// TODO: find a way to use Iterable here instead of List (Iterable is not accepted by json serializable)
List<TreeNode> fromTreeNodesJson(Iterable<Map<String, dynamic>> collection) =>
    collection?.map((nodeJson) => TreeNode.fromJson(nodeJson))?.toList();
