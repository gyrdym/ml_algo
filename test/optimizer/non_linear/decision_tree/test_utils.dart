import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_leaf_label.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_node.dart';
import 'package:ml_linalg/vector.dart';
import 'package:test/test.dart';
import 'package:xrange/zrange.dart';

void testTreeNode(
    DecisionTreeNode node,
    {
      bool shouldBeLeaf,
      double expectedSplittingNumericalValue,
      ZRange expectedSplittingColumnRange,
      Vector expectedSplittingNominalValue,
      int expectedChildrenLength,
      DecisionTreeLeafLabel expectedLabel,
    }
) {
  expect(node.isLeaf, equals(shouldBeLeaf));
  expect(node.splittingNumericalValue, equals(expectedSplittingNumericalValue));
  expect(node.splittingColumnRange, equals(expectedSplittingColumnRange));
  expect(node.splittingNominalValue, equals(expectedSplittingNominalValue));
  expectedChildrenLength == null
      ? expect(node.children, isNull)
      : expect(node.children, hasLength(expectedChildrenLength));
  expectedLabel == null
      ? expect(node.label, isNull)
      : testLeafLabel(node.label, expectedLabel);
}

void testLeafLabel(DecisionTreeLeafLabel label,
    DecisionTreeLeafLabel expectedLabel) {
  expect(label.nominalValue, equals(expectedLabel.nominalValue));
  expect(label.numericalValue, equals(expectedLabel.numericalValue));
  expect(label.probability, equals(expectedLabel.probability));
}