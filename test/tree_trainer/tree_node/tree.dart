import 'package:ml_algo/src/tree_trainer/tree_node/split_predicate/predicate_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';

final _child31 = TreeNode(
  PredicateType.equalTo,
  600,
  null,
  [],
  null,
);

final _child32 = TreeNode(
  PredicateType.equalTo,
  700,
  null,
  [],
  null,
);

final _child33 = TreeNode(
  PredicateType.equalTo,
  800,
  null,
  [],
  null,
);

final _child34 = TreeNode(
  PredicateType.equalTo,
  900,
  null,
  [],
  null,
);

final _child35 = TreeNode(
  PredicateType.equalTo,
  900,
  null,
  [],
  null,
);

final _child36 = TreeNode(
  PredicateType.equalTo,
  901,
  null,
  [],
  null,
);

final _child37 = TreeNode(
  PredicateType.equalTo,
  911,
  null,
  [],
  null,
);

final _child21 = TreeNode(
  PredicateType.equalTo,
  100,
  null,
  [
    _child31,
    _child32,
  ],
  null,
);

final _child22 = TreeNode(
  PredicateType.equalTo,
  200,
  null,
  [
    _child33,
    _child34,
  ],
  null,
);

final _child23 = TreeNode(
  PredicateType.equalTo,
  300,
  null,
  [
    _child35,
    _child36,
  ],
  null,
);

final _child24 = TreeNode(
  PredicateType.equalTo,
  400,
  null,
  [
    _child37,
  ],
  null,
);

final _child25 = TreeNode(
  PredicateType.equalTo,
  500,
  null,
  [],
  null,
);

final _child11 = TreeNode(
  PredicateType.equalTo,
  10,
  null,
  [_child21, _child22],
  null,
);

final _child12 = TreeNode(
  PredicateType.equalTo,
  12,
  null,
  [
    _child23,
    _child24,
  ],
  null,
);

final _child13 = TreeNode(
  PredicateType.equalTo,
  13,
  null,
  [
    _child25,
  ],
  null,
);

final tree = TreeNode(
  PredicateType.equalTo,
  null,
  null,
  [
    _child11,
    _child12,
    _child13,
  ],
  null,
);
