import 'package:ml_algo/src/optimizer/non_linear/decision_tree/best_stump_finder/best_stump_finder.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_leaf_label.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_node.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_stump.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_detector/leaf_detector.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';
import 'package:xrange/zrange.dart';

import '../../../test_utils/mocks.dart';

void main() {
  group('DecisionTreeOptimizer', () {
    test('should build a greedy classifier tree', () {
      final observations = Matrix.fromList([
        [10, 20, 30, 40, 0, 0, 1],
        [90, 51, 34, 31, 0, 0, 1],
        [23, 40, 90, 50, 0, 1, 0],
        [55, 10, 22, 80, 1, 0, 0],
      ]);

      final featuresColumnRange = [
        ZRange.singleton(0),
        ZRange.singleton(1),
        ZRange.singleton(2),
        ZRange.singleton(3),
      ];

      final outcomesColumnRange = ZRange.closed(4, 6);

      final rangeToCategoricalValues = {
        outcomesColumnRange: [
          Vector.fromList([0, 0, 1]),
          Vector.fromList([0, 1, 0]),
          Vector.fromList([1, 0, 0]),
        ],
      };

      final leafDetector = LeafDetectorMock();
      final leafLabelFactory = LeafLabelFactoryMock();
      final bestStumpFinder = BestStumpFinderMock();

      final mockIsLeafFnCall = (Matrix samples, bool isLeaf) =>
          mockLeafDetectorCall(leafDetector, samples, outcomesColumnRange,
              isLeaf);

      final mockFindBestStumpFnCall = (Matrix input, {double expectedSplitValue,
          ZRange expectedSplitRange, List<Matrix> expectedOutput}) =>
          mockStumpFinderCall(bestStumpFinder, input, outcomesColumnRange,
              featuresColumnRange, rangeToCategoricalValues, expectedOutput,
              expectedSplitValue, null, expectedSplitRange);

      mockIsLeafFnCall(observations, false);

      mockFindBestStumpFnCall(observations,
          expectedSplitValue: 34,
          expectedSplitRange: ZRange.singleton(2),
          expectedOutput: [
            Matrix.fromList([
              [10, 20, 30, 40, 0, 0, 1],
            ]),
            Matrix.fromList([
              [90, 51, 34, 31, 0, 0, 1],
              [23, 40, 90, 50, 0, 1, 0],
              [55, 10, 22, 80, 1, 0, 0],
            ]),
          ]
      );

      mockIsLeafFnCall(Matrix.fromList([
        [10, 20, 30, 40, 0, 0, 1]
      ]), true);

      mockIsLeafFnCall(Matrix.fromList([
        [90, 51, 34, 31, 0, 0, 1],
        [23, 40, 90, 50, 0, 1, 0],
        [55, 10, 22, 80, 1, 0, 0],
      ]), false);

      mockFindBestStumpFnCall(
        Matrix.fromList([
          [90, 51, 34, 31, 0, 0, 1],
          [23, 40, 90, 50, 0, 1, 0],
          [55, 10, 22, 80, 1, 0, 0],
        ]),
        expectedSplitValue: 50,
        expectedSplitRange: ZRange.singleton(3),
        expectedOutput: [
          Matrix.fromList([
            [90, 51, 34, 31, 0, 0, 1],
            [23, 40, 90, 50, 0, 1, 0],
          ]),
          Matrix.fromList([
            [55, 10, 22, 80, 1, 0, 0],
          ]),
        ],
      );

      mockIsLeafFnCall(Matrix.fromList([
        [90, 51, 34, 31, 0, 0, 1],
        [23, 40, 90, 50, 0, 1, 0],
      ]), false);

      mockFindBestStumpFnCall(
        Matrix.fromList([
          [90, 51, 34, 31, 0, 0, 1],
          [23, 40, 90, 50, 0, 1, 0],
        ]),
        expectedSplitValue: 90,
        expectedSplitRange: ZRange.singleton(0),
        expectedOutput: [
          Matrix.fromList([
            [90, 51, 34, 31, 0, 0, 1],
          ]),
          Matrix.fromList([
            [23, 40, 90, 50, 0, 1, 0],
          ]),
        ],
      );

      mockIsLeafFnCall(Matrix.fromList([
        [90, 51, 34, 31, 0, 0, 1],
      ]), true);

      mockIsLeafFnCall(Matrix.fromList([
        [23, 40, 90, 50, 0, 1, 0],
      ]), true);

      mockIsLeafFnCall(Matrix.fromList([
        [55, 10, 22, 80, 1, 0, 0],
      ]), true);

      final rootNode = DecisionTreeOptimizer(
          observations,
          featuresColumnRange,
          outcomesColumnRange,
          rangeToCategoricalValues,
          leafDetector,
          leafLabelFactory,
          bestStumpFinder,
      ).root;

      testTreeNode(rootNode,
          shouldBeLeaf: false,
          expectedSplittingNumericalValue: 34.0,
          expectedSplittingColumnRange: ZRange.singleton(2),
          expectedSplittingNominalValues: null,
          expectedChildrenLength: 2,
          expectedLabel: null
      );

      final firstLevelNodes = rootNode.children;

      testTreeNode(firstLevelNodes.first,
          shouldBeLeaf: true,
          expectedSplittingNumericalValue: null,
          expectedSplittingColumnRange: null,
          expectedSplittingNominalValues: null,
          expectedChildrenLength: null,
          expectedLabel: null,
      );

      testTreeNode(firstLevelNodes.last,
          shouldBeLeaf: false,
          expectedSplittingNumericalValue: 50,
          expectedSplittingColumnRange: ZRange.singleton(3),
          expectedSplittingNominalValues: null,
          expectedChildrenLength: 2,
          expectedLabel: null
      );

      final secondLevelNodes = firstLevelNodes.last.children;

      testTreeNode(secondLevelNodes.first,
          shouldBeLeaf: false,
          expectedSplittingNumericalValue: 90,
          expectedSplittingColumnRange: ZRange.singleton(0),
          expectedSplittingNominalValues: null,
          expectedChildrenLength: 2,
          expectedLabel: null
      );
    });
  });
}

void mockLeafDetectorCall(
    LeafDetector leafDetector,
    Matrix split,
    ZRange outcomesColumnRange,
    bool returnValue,
) {
  when(leafDetector.isLeaf(
    argThat(equals(split)),
    outcomesColumnRange,
  )).thenReturn(returnValue);
}

void mockStumpFinderCall(
    BestStumpFinder bestStumpFinder,
    Matrix input,
    ZRange outcomesColumnRange,
    List<ZRange> featuresColumnRange,
    Map<ZRange, List<Vector>> rangeToCategoricalValues,
    List<Matrix> expectedOutput,
    double expectedSplittingValue,
    List<Vector> expectedSplittingNominalValues,
    ZRange expectedSplittingRange,
) {
  when(bestStumpFinder.find(
    argThat(equals(input)),
    outcomesColumnRange,
    featuresColumnRange,
    rangeToCategoricalValues,
  )).thenReturn(
    DecisionTreeStump(
      expectedSplittingValue,
      expectedSplittingNominalValues,
      expectedSplittingRange,
      expectedOutput,
    ),
  );
}

void testTreeNode(
    DecisionTreeNode node,
    {
      bool shouldBeLeaf,
      double expectedSplittingNumericalValue,
      ZRange expectedSplittingColumnRange,
      List<Vector> expectedSplittingNominalValues,
      int expectedChildrenLength,
      DecisionTreeLeafLabel expectedLabel,
    }
) {
  expect(node.isLeaf, equals(shouldBeLeaf));
  expect(node.splittingNumericalValue, equals(expectedSplittingNumericalValue));
  expect(node.splittingColumnRange, equals(expectedSplittingColumnRange));
  expect(node.splittingNominalValues, equals(expectedSplittingNominalValues));
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
