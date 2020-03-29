import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/tree_node_splitting_predicate_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node_serialize.dart';
import 'package:ml_tech/unit_testing/readers/json.dart';
import 'package:test/test.dart';

void main() {
  group('TreeNode', () {
    test('should serialize itself', () async {
      final child31 = TreeNode(
        TreeNodeSplittingPredicateType.equalTo,
        600,
        null,
        [],
        null,
        3,
      );

      final child32 = TreeNode(
        TreeNodeSplittingPredicateType.equalTo,
        700,
        null,
        [],
        null,
        3,
      );

      final child33 = TreeNode(
        TreeNodeSplittingPredicateType.equalTo,
        800,
        null,
        [],
        null,
        3,
      );

      final child34 = TreeNode(
        TreeNodeSplittingPredicateType.equalTo,
        900,
        null,
        [],
        null,
        3,
      );

      final child35 = TreeNode(
        TreeNodeSplittingPredicateType.equalTo,
        900,
        null,
        [],
        null,
        3,
      );

      final child36 = TreeNode(
        TreeNodeSplittingPredicateType.equalTo,
        901,
        null,
        [],
        null,
        3,
      );

      final child37 = TreeNode(
        TreeNodeSplittingPredicateType.equalTo,
        911,
        null,
        [],
        null,
        3,
      );

      final child21 = TreeNode(
        TreeNodeSplittingPredicateType.equalTo,
        100,
        null,
        [
          child31,
          child32,
        ],
        null,
        2,
      );

      final child22 = TreeNode(
        TreeNodeSplittingPredicateType.equalTo,
        200,
        null,
        [
          child33,
          child34,
        ],
        null,
        2,
      );

      final child23 = TreeNode(
        TreeNodeSplittingPredicateType.equalTo,
        300,
        null,
        [
          child35,
          child36,
        ],
        null,
        2,
      );

      final child24 = TreeNode(
        TreeNodeSplittingPredicateType.equalTo,
        400,
        null,
        [
          child37,
        ],
        null,
        2,
      );

      final child25 = TreeNode(
        TreeNodeSplittingPredicateType.equalTo,
        500,
        null,
        [],
        null,
        2,
      );

      final child11 = TreeNode(
        TreeNodeSplittingPredicateType.equalTo,
        10,
        null,
        [
          child21,
          child22
        ],
        null,
        1,
      );

      final child12 = TreeNode(
        TreeNodeSplittingPredicateType.equalTo,
        12,
        null,
        [
          child23,
          child24,
        ],
        null,
        1,
      );

      final child13 = TreeNode(
        TreeNodeSplittingPredicateType.equalTo,
        13,
        null,
        [
          child25,
        ],
        null,
        1,
      );

      final root = TreeNode(
        TreeNodeSplittingPredicateType.equalTo,
        null,
        null,
        [
          child11,
          child12,
          child13,
        ],
        null,
      );

      final snapshotFileName = 'test/tree_trainer/tree_node_test.json';
      final actual = serialize(root);
      final expected = await readJSON(snapshotFileName);

      expect(actual, expected);
    });
  });
}
