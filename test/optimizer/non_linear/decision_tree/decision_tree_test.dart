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
      final mockIsLeafFn = (Matrix samples, bool isLeaf) =>
          mockLeafDetectorCall(leafDetector, samples, outcomesColumnRange,
              isLeaf);
      final mockFindBestStumpFn = (Matrix input, double splitValue,
          ZRange splitRange, List<Matrix> output) =>
          mockStumpFinderCall(bestStumpFinder, input, outcomesColumnRange,
              featuresColumnRange, rangeToCategoricalValues, output, splitValue,
              null, splitRange);

      mockIsLeafFn(observations, false);

      mockFindBestStumpFn(observations, 34, ZRange.singleton(2), [
        Matrix.fromList([
          [10, 20, 30, 40, 0, 0, 1],
        ]),
        Matrix.fromList([
          [90, 51, 34, 31, 0, 0, 1],
          [23, 40, 90, 50, 0, 1, 0],
          [55, 10, 22, 80, 1, 0, 0],
        ]),
      ]);

      mockIsLeafFn(Matrix.fromList([
        [10, 20, 30, 40, 0, 0, 1]
      ]), true);

      mockIsLeafFn(Matrix.fromList([
        [90, 51, 34, 31, 0, 0, 1],
        [23, 40, 90, 50, 0, 1, 0],
        [55, 10, 22, 80, 1, 0, 0],
      ]), false);

      mockFindBestStumpFn(
        Matrix.fromList([
          [90, 51, 34, 31, 0, 0, 1],
          [23, 40, 90, 50, 0, 1, 0],
          [55, 10, 22, 80, 1, 0, 0],
        ]),
        50,
        ZRange.singleton(3),
        [
          Matrix.fromList([
            [90, 51, 34, 31, 0, 0, 1],
            [23, 40, 90, 50, 0, 1, 0],
          ]),
          Matrix.fromList([
            [55, 10, 22, 80, 1, 0, 0],
          ]),
        ],
      );

      mockIsLeafFn(Matrix.fromList([
        [90, 51, 34, 31, 0, 0, 1],
        [23, 40, 90, 50, 0, 1, 0],
      ]), false);

      mockFindBestStumpFn(
        Matrix.fromList([
          [90, 51, 34, 31, 0, 0, 1],
          [23, 40, 90, 50, 0, 1, 0],
        ]),
        90,
        ZRange.singleton(0),
        [
          Matrix.fromList([
            [90, 51, 34, 31, 0, 0, 1],
          ]),
          Matrix.fromList([
            [23, 40, 90, 50, 0, 1, 0],
          ]),
        ],
      );

      mockIsLeafFn(Matrix.fromList([
        [90, 51, 34, 31, 0, 0, 1],
      ]), true);

      mockIsLeafFn(Matrix.fromList([
        [23, 40, 90, 50, 0, 1, 0],
      ]), true);

      mockIsLeafFn(Matrix.fromList([
        [55, 10, 22, 80, 1, 0, 0],
      ]), true);

      final tree = DecisionTreeOptimizer(
          observations,
          featuresColumnRange,
          outcomesColumnRange,
          rangeToCategoricalValues,
          leafDetector,
          leafLabelFactory,
          bestStumpFinder,
      ).root;

      testTreeNode(tree, false, 34.0, ZRange.singleton(2), null, 2, null);

      final firstLevel = tree.children;

      testTreeNode(firstLevel.first, true, null, null, null, null, null);
      testTreeNode(firstLevel.last, false, 50, ZRange.singleton(3), null, 2,
          null);

      final secondLevelRight = firstLevel.last.children;

      testTreeNode(secondLevelRight.first, false, 90, ZRange.singleton(0), null,
          2, null);
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
    bool isLeaf,
    double splittingNumericalValue,
    ZRange splittingColumnRange,
    List<Vector> splittingNominalValues,
    int childrenLength,
    DecisionTreeLeafLabel expectedLabel,
) {
  expect(node.isLeaf, equals(isLeaf));
  expect(node.splittingNumericalValue, equals(splittingNumericalValue));
  expect(node.splittingColumnRange, equals(splittingColumnRange));
  expect(node.splittingNominalValues, equals(splittingNominalValues));
  childrenLength == null
      ? expect(node.children, isNull)
      : expect(node.children, hasLength(childrenLength));
  expectedLabel == null
      ? expect(node.label, isNull)
      : testLeafLabel(node.label, expectedLabel);
}

void testLeafLabel(DecisionTreeLeafLabel label, DecisionTreeLeafLabel expectedLabel) {

}
