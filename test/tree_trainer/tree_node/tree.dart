import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/tree_node_splitting_predicate_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';

final _child31 = TreeNode(
  TreeNodeSplittingPredicateType.equalTo,
  600,
  null,
  [],
  null,
  3,
);

final _child32 = TreeNode(
  TreeNodeSplittingPredicateType.equalTo,
  700,
  null,
  [],
  null,
  3,
);

final _child33 = TreeNode(
  TreeNodeSplittingPredicateType.equalTo,
  800,
  null,
  [],
  null,
  3,
);

final _child34 = TreeNode(
  TreeNodeSplittingPredicateType.equalTo,
  900,
  null,
  [],
  null,
  3,
);

final _child35 = TreeNode(
  TreeNodeSplittingPredicateType.equalTo,
  900,
  null,
  [],
  null,
  3,
);

final _child36 = TreeNode(
  TreeNodeSplittingPredicateType.equalTo,
  901,
  null,
  [],
  null,
  3,
);

final _child37 = TreeNode(
  TreeNodeSplittingPredicateType.equalTo,
  911,
  null,
  [],
  null,
  3,
);

final _child21 = TreeNode(
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

final _child22 = TreeNode(
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

final _child23 = TreeNode(
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

final _child24 = TreeNode(
  TreeNodeSplittingPredicateType.equalTo,
  400,
  null,
  [
    _child37,
  ],
  null,
  2,
);

final _child25 = TreeNode(
  TreeNodeSplittingPredicateType.equalTo,
  500,
  null,
  [],
  null,
  2,
);

final _child11 = TreeNode(
  TreeNodeSplittingPredicateType.equalTo,
  10,
  null,
  [_child21, _child22],
  null,
  1,
);

final _child12 = TreeNode(
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

final _child13 = TreeNode(
  TreeNodeSplittingPredicateType.equalTo,
  13,
  null,
  [
    _child25,
  ],
  null,
  1,
);

final tree = TreeNode(
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
