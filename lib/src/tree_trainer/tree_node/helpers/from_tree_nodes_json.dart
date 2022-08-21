import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';

List<TreeNode>? fromTreeNodesJson(Iterable<dynamic>? collection) => collection
    ?.map((dynamic nodeJson) =>
        TreeNode.fromJson(nodeJson as Map<String, dynamic>))
    .toList();
