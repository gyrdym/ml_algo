import 'package:ml_algo/src/tree_trainer/tree_node/decision_tree_node/decision_tree_node.dart';

List<DecisionTreeNode>? fromTreeNodesJson(Iterable? collection) => collection
    ?.map((dynamic nodeJson) =>
        DecisionTreeNode.fromJson(nodeJson as Map<String, dynamic>))
    .toList();
