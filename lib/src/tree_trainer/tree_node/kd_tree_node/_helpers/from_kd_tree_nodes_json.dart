import 'package:ml_algo/src/tree_trainer/tree_node/kd_tree_node/kd_tree_node.dart';

List<KDTreeNode>? fromKDTreeNodesJson(Iterable? collection) => collection
    ?.map((dynamic nodeJson) =>
        KDTreeNode.fromJson(nodeJson as Map<String, dynamic>))
    .toList();
