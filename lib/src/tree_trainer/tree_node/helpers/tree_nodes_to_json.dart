import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';

Iterable<Map<String, dynamic>>? treeNodesToJson(
        Iterable<TreeNode>? collection) =>
    collection?.map((node) => node.toJson()).toList();
