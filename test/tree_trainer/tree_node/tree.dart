import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/tree_node_splitting_predicate_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/decision_tree_node/decision_tree_node.dart';

final _child31 = DecisionTreeNode(
  TreeNodeSplittingPredicateType.equalTo,
  600,
  null,
  [],
  null,
  3,
);

final _child32 = DecisionTreeNode(
  TreeNodeSplittingPredicateType.equalTo,
  700,
  null,
  [],
  null,
  3,
);

final _child33 = DecisionTreeNode(
  TreeNodeSplittingPredicateType.equalTo,
  800,
  null,
  [],
  null,
  3,
);

final _child34 = DecisionTreeNode(
  TreeNodeSplittingPredicateType.equalTo,
  900,
  null,
  [],
  null,
  3,
);

final _child35 = DecisionTreeNode(
  TreeNodeSplittingPredicateType.equalTo,
  900,
  null,
  [],
  null,
  3,
);

final _child36 = DecisionTreeNode(
  TreeNodeSplittingPredicateType.equalTo,
  901,
  null,
  [],
  null,
  3,
);

final _child37 = DecisionTreeNode(
  TreeNodeSplittingPredicateType.equalTo,
  911,
  null,
  [],
  null,
  3,
);

final _child21 = DecisionTreeNode(
  TreeNodeSplittingPredicateType.equalTo,
  100,
  null,
  [
    _child31,
    _child32,
  ],
  null,
  2,
);

final _child22 = DecisionTreeNode(
  TreeNodeSplittingPredicateType.equalTo,
  200,
  null,
  [
    _child33,
    _child34,
  ],
  null,
  2,
);

final _child23 = DecisionTreeNode(
  TreeNodeSplittingPredicateType.equalTo,
  300,
  null,
  [
    _child35,
    _child36,
  ],
  null,
  2,
);

final _child24 = DecisionTreeNode(
  TreeNodeSplittingPredicateType.equalTo,
  400,
  null,
  [
    _child37,
  ],
  null,
  2,
);

final _child25 = DecisionTreeNode(
  TreeNodeSplittingPredicateType.equalTo,
  500,
  null,
  [],
  null,
  2,
);

final _child11 = DecisionTreeNode(
  TreeNodeSplittingPredicateType.equalTo,
  10,
  null,
  [_child21, _child22],
  null,
  1,
);

final _child12 = DecisionTreeNode(
  TreeNodeSplittingPredicateType.equalTo,
  12,
  null,
  [
    _child23,
    _child24,
  ],
  null,
  1,
);

final _child13 = DecisionTreeNode(
  TreeNodeSplittingPredicateType.equalTo,
  13,
  null,
  [
    _child25,
  ],
  null,
  1,
);

final tree = DecisionTreeNode(
  TreeNodeSplittingPredicateType.equalTo,
  null,
  null,
  [
    _child11,
    _child12,
    _child13,
  ],
  null,
);
