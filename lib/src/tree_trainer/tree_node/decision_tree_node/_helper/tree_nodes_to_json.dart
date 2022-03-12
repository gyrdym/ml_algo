import 'package:ml_algo/src/tree_trainer/tree_node/decision_tree_node/decision_tree_node.dart';

Iterable<Map<String, dynamic>>? treeNodesToJson(
        Iterable<DecisionTreeNode>? collection) =>
    collection?.map((node) => node.toJson()).toList();
