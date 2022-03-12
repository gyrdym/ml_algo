import 'package:ml_algo/src/tree_trainer/tree_node/kd_tree_node/kd_tree_node.dart';

Iterable<Map<String, dynamic>>? kdTreeNodesToJson(
        Iterable<KDTreeNode>? collection) =>
    collection?.map((node) => node.toJson()).toList();
