import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/split_predicate/predicate_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';
import 'package:test/test.dart';

import '../../helpers.dart';
import 'tree.dart';

void main() {
  group('TreeNode', () {
    final snapshotFileName = 'test/tree_trainer/tree_node/tree_node_test.json';
    final predicateType = PredicateType.equalTo;
    final splittingValue = 600;
    final splittingIndex = 2;
    final children = <TreeNode>[];
    final treeLabel = TreeLeafLabel(1000, probability: 0.75);

    final node = TreeNode(
      predicateType,
      splittingValue,
      splittingIndex,
      children,
      treeLabel,
    );

    test('should hold splitting predicate type', () {
      expect(node.predicateType, predicateType);
    });

    test('should hold splitting value', () {
      expect(node.splitValue, splittingValue);
    });

    test('should hold splitting index', () {
      expect(node.splitIndex, splittingIndex);
    });

    test('should hold child node list', () {
      expect(node.children, children);
    });

    test('should hold tree leaf label', () {
      expect(node.label, treeLabel);
    });

    test('should serialize', () async {
      final actual = tree.toJson();
      final expected = await readJSON(snapshotFileName);

      expect(actual, expected);
    });

    test('should restore from json', () async {
      final json = await readJSON(snapshotFileName);
      final actual = TreeNode.fromJson(json);

      expect(actual.toJson(), tree.toJson());
    });

    test('should return correct shape of the tree', () async {
      final json = await readJSON(snapshotFileName);
      final node = TreeNode.fromJson(json);

      expect(node.shape, {
        0: 3,
        1: 5,
        2: 7,
        3: 0,
      });
    });

    test('should return correct shape of the tree in case of empty node', () {
      expect(node.shape, {0: 0});
    });
  });
}
