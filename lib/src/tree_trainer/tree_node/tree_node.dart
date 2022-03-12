import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/tree_node_splitting_predicate_type.dart';

abstract class TreeNode {
  List<TreeNode>? get children;
  TreeNodeSplittingPredicateType? get predicateType;
  num? get splittingValue;
  int? get splittingIndex;
  int get level;
}
