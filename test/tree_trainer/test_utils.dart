import 'package:ml_algo/src/tree_trainer/tree_node/decision_tree_node/decision_tree_node.dart';
import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label.dart';
import 'package:ml_linalg/vector.dart';
import 'package:test/test.dart';

void testTreeNode(
  DecisionTreeNode node, {
  required bool shouldBeLeaf,
  required double expectedSplittingValue,
  required int expectedSplittingColumnIdx,
  int? expectedChildrenLength,
  TreeLeafLabel? expectedLabel,
  Map<Vector, bool>? samplesToCheck,
}) {
  expect(node.isLeaf, equals(shouldBeLeaf),
      reason: 'node.isLeaf should be $shouldBeLeaf');

  expect(node.splittingValue, equals(expectedSplittingValue));
  expect(node.splittingIndex, equals(expectedSplittingColumnIdx));

  expectedChildrenLength == null
      ? expect(node.children, isNull)
      : expect(node.children, hasLength(expectedChildrenLength));

  expectedLabel == null
      ? expect(node.label, isNull)
      : testLeafLabel(node.label!, expectedLabel);

  samplesToCheck?.entries.forEach((entry) {
    expect(node.isSamplePassed(entry.key), entry.value,
        reason: '`isSamplePassed` should return ${entry.value} for '
            'sampe ${entry.key}');
  });
}

void testLeafLabel(TreeLeafLabel label, TreeLeafLabel expectedLabel) {
  expect(label.value, equals(expectedLabel.value));
  expect(label.probability, equals(expectedLabel.probability));
}
