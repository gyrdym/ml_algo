import 'package:ml_algo/src/optimizer/non_linear/decision_tree/split_selector/split_selector.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_solver.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_leaf_label.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_node.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_detector/leaf_detector.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_label_factory/leaf_label_factory.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';
import 'package:xrange/zrange.dart';

import '../../../test_utils/mocks.dart';
import 'test_utils.dart';

void main() {
  group('DecisionTreeSolver', () {
    test('should build a greedy classifier tree', () {
      final observations = Matrix.fromList([
        [10, 20, 30, 40, 0, 0, 1],
        [90, 51, 34, 31, 0, 0, 1],
        [23, 40, 90, 50, 0, 1, 0],
        [55, 10, 22, 80, 1, 0, 0],
      ]);

      final featuresColumnRanges = Set<ZRange>.from(<ZRange>[
        ZRange.singleton(0),
        ZRange.singleton(1),
        ZRange.singleton(2),
        ZRange.singleton(3),
      ]);

      final outcomesColumnRange = ZRange.closed(4, 6);

      final rangeToNominalValues = {
        outcomesColumnRange: [
          Vector.fromList([0, 0, 1]),
          Vector.fromList([0, 1, 0]),
          Vector.fromList([1, 0, 0]),
        ],
      };

      final leafDetector = LeafDetectorMock();
      final leafLabelFactory = LeafLabelFactoryMock();
      final bestStumpFinder = BestSplitFinderMock();

      final mockIsLeafFnCall = (Matrix samples, bool isLeaf) =>
          mockLeafDetectorCall(leafDetector, samples, outcomesColumnRange,
              featuresColumnRanges, isLeaf);

      final mockLeafLabelFactoryFnCall = ({Matrix leafObservations,
        DecisionTreeLeafLabel expectedLabel}) =>
          mockLeafLabelFactoryCall(leafLabelFactory: leafLabelFactory,
              leafObservations: leafObservations,
              outcomesColumnRange: outcomesColumnRange,
              isClassLabelNominal: true,
              expectedLabel: expectedLabel);

      final mockFindBestStumpFnCall = (Matrix input, {double expectedSplitValue,
          ZRange expectedSplitRange, List<Matrix> expectedOutput}) =>
          mockStumpFinderCall(bestStumpFinder, input, outcomesColumnRange,
              featuresColumnRanges, rangeToNominalValues, expectedOutput,
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

      mockLeafLabelFactoryFnCall(
          leafObservations: Matrix.fromList([
            [10, 20, 30, 40, 0, 0, 1]
          ]),
          expectedLabel: DecisionTreeLeafLabel
              .nominal(Vector.fromList([0, 0, 1])),
      );

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

      mockLeafLabelFactoryFnCall(
        leafObservations: Matrix.fromList([
          [90, 51, 34, 31, 0, 0, 1],
        ]),
        expectedLabel: DecisionTreeLeafLabel
            .nominal(Vector.fromList([0, 0, 1])),
      );

      mockIsLeafFnCall(Matrix.fromList([
        [23, 40, 90, 50, 0, 1, 0],
      ]), true);

      mockLeafLabelFactoryFnCall(
        leafObservations: Matrix.fromList([
          [23, 40, 90, 50, 0, 1, 0],
        ]),
        expectedLabel: DecisionTreeLeafLabel
            .nominal(Vector.fromList([0, 1, 0])),
      );

      mockIsLeafFnCall(Matrix.fromList([
        [55, 10, 22, 80, 1, 0, 0],
      ]), true);

      mockLeafLabelFactoryFnCall(
        leafObservations: Matrix.fromList([
          [55, 10, 22, 80, 1, 0, 0],
        ]),
        expectedLabel: DecisionTreeLeafLabel
            .nominal(Vector.fromList([1, 0, 0])),
      );

      final rootNode = DecisionTreeSolver(
          observations,
          featuresColumnRanges,
          outcomesColumnRange,
          rangeToNominalValues,
          leafDetector,
          leafLabelFactory,
          bestStumpFinder,
      ).root;

      testTreeNode(rootNode,
          shouldBeLeaf: false,
          expectedSplittingNumericalValue: 34.0,
          expectedSplittingColumnRange: ZRange.singleton(2),
          expectedSplittingNominalValue: null,
          expectedChildrenLength: 2,
          expectedLabel: null,
      );

      final firstLevelNodes = rootNode.children;
      final secondLevelNodes = firstLevelNodes.last.children;
      final thirdLevelNodes = secondLevelNodes.first.children;

      testTreeNode(firstLevelNodes.first,
          shouldBeLeaf: true,
          expectedSplittingNumericalValue: null,
          expectedSplittingColumnRange: null,
          expectedSplittingNominalValue: null,
          expectedChildrenLength: null,
          expectedLabel: DecisionTreeLeafLabel
              .nominal(Vector.fromList([0, 0, 1])),
      );

      testTreeNode(firstLevelNodes.last,
          shouldBeLeaf: false,
          expectedSplittingNumericalValue: 50,
          expectedSplittingColumnRange: ZRange.singleton(3),
          expectedSplittingNominalValue: null,
          expectedChildrenLength: 2,
          expectedLabel: null
      );

      testTreeNode(secondLevelNodes.first,
          shouldBeLeaf: false,
          expectedSplittingNumericalValue: 90,
          expectedSplittingColumnRange: ZRange.singleton(0),
          expectedSplittingNominalValue: null,
          expectedChildrenLength: 2,
          expectedLabel: null
      );

      testTreeNode(secondLevelNodes.last,
          shouldBeLeaf: true,
          expectedSplittingNumericalValue: null,
          expectedSplittingColumnRange: null,
          expectedSplittingNominalValue: null,
          expectedChildrenLength: null,
          expectedLabel: DecisionTreeLeafLabel
              .nominal(Vector.fromList([1, 0, 0])),
      );

      testTreeNode(thirdLevelNodes.first,
        shouldBeLeaf: true,
        expectedSplittingNumericalValue: null,
        expectedSplittingColumnRange: null,
        expectedSplittingNominalValue: null,
        expectedChildrenLength: null,
        expectedLabel: DecisionTreeLeafLabel
            .nominal(Vector.fromList([0, 0, 1])),
      );

      testTreeNode(thirdLevelNodes.last,
        shouldBeLeaf: true,
        expectedSplittingNumericalValue: null,
        expectedSplittingColumnRange: null,
        expectedSplittingNominalValue: null,
        expectedChildrenLength: null,
        expectedLabel: DecisionTreeLeafLabel
            .nominal(Vector.fromList([0, 1, 0])),
      );
    });

    test('should exclude nominal feature range from the features column ranges '
        'after splitting by this feature', () {
      final samples = Matrix.fromList([
        [10, 1, 0, 0, 0, 0, 1],
        [90, 1, 0, 0, 0, 1, 0],
        [23, 0, 0, 1, 0, 1, 0],
        [55, 0, 1, 0, 1, 0, 0],
      ]);

      final nominalFeatureRange = ZRange.closed(1, 3);
      final featuresColumnRangesFull = Set<ZRange>.from(<ZRange>[
        ZRange.singleton(0),
        nominalFeatureRange,
      ]);
      final featuresColumnRangesReduced = Set<ZRange>.from(<ZRange>[
        ZRange.singleton(0),
      ]);

      final outcomesColumnRange = ZRange.closed(4, 6);

      final nominalFeatureValues = [
        Vector.fromList([0, 0, 1]),
        Vector.fromList([0, 1, 0]),
        Vector.fromList([1, 0, 0]),
      ];

      final nominalOutcomeValues = [
        Vector.fromList([0, 0, 1]),
        Vector.fromList([0, 1, 0]),
        Vector.fromList([1, 0, 0]),
      ];

      final rangeToNominalValues = {
        nominalFeatureRange: nominalFeatureValues,
        outcomesColumnRange: nominalOutcomeValues,
      };

      final leafDetector = LeafDetectorMock();
      final leafLabelFactory = LeafLabelFactoryMock();
      final bestStumpFinder = BestSplitFinderMock();

      final mockBoundIsLeafFnCall = (Matrix samples,
          Iterable<ZRange> featuresRanges, bool isLeaf) =>
          mockLeafDetectorCall(leafDetector, samples, outcomesColumnRange,
              featuresRanges, isLeaf);

      final mockBoundLeafLabelFactoryFnCall = ({Matrix leafObservations,
        DecisionTreeLeafLabel expectedLabel}) =>
          mockLeafLabelFactoryCall(leafLabelFactory: leafLabelFactory,
              leafObservations: leafObservations,
              outcomesColumnRange: outcomesColumnRange,
              isClassLabelNominal: true,
              expectedLabel: expectedLabel);

      final mockBoundFindBestStumpFnCall = (Matrix input, {
        Set<ZRange> featuresColumnRanges, double expectedSplitValue,
        List<Vector> expectedSplitNominalValues, ZRange expectedSplitRange,
        List<Matrix> expectedOutput}) =>
          mockStumpFinderCall(bestStumpFinder, input, outcomesColumnRange,
              featuresColumnRanges, rangeToNominalValues, expectedOutput,
              expectedSplitValue, expectedSplitNominalValues,
              expectedSplitRange);

      mockBoundIsLeafFnCall(samples, featuresColumnRangesFull, false);
      mockBoundFindBestStumpFnCall(samples,
          featuresColumnRanges: featuresColumnRangesFull,
          expectedSplitValue: null,
          expectedSplitNominalValues: nominalFeatureValues,
          expectedSplitRange: nominalFeatureRange,
          expectedOutput: [
            Matrix.fromList([
              [10, 1, 0, 0, 0, 1, 0],
              [90, 1, 0, 0, 0, 0, 1],
            ]),
            Matrix.fromList([
              [23, 0, 0, 1, 0, 1, 0],
            ]),
            Matrix.fromList([
              [55, 0, 1, 0, 1, 0, 0],
            ]),
          ],
      );

      mockBoundIsLeafFnCall(Matrix.fromList([
        [10, 1, 0, 0, 0, 1, 0],
        [90, 1, 0, 0, 0, 0, 1],
      ]), featuresColumnRangesReduced, false);
      mockBoundFindBestStumpFnCall(Matrix.fromList([
          [10, 1, 0, 0, 0, 1, 0],
          [90, 1, 0, 0, 0, 0, 1],
        ]),
        featuresColumnRanges: Set.from(<ZRange>[ZRange.singleton(0)]),
        expectedSplitValue: 10,
        expectedSplitNominalValues: null,
        expectedSplitRange: ZRange.singleton(0),
        expectedOutput: [
          Matrix.fromList([
            [10, 1, 0, 0, 0, 1, 0],
          ]),
          Matrix.fromList([
            [90, 1, 0, 0, 0, 0, 1],
          ]),
        ],
      );

      mockBoundIsLeafFnCall(Matrix.fromList([
        [10, 1, 0, 0, 0, 1, 0],
      ]), featuresColumnRangesReduced, true);
      mockBoundLeafLabelFactoryFnCall(
        leafObservations: Matrix.fromList([
          [10, 1, 0, 0, 0, 1, 0],
        ]),
        expectedLabel: DecisionTreeLeafLabel
            .nominal(Vector.fromList([0, 1, 0])),
      );

      mockBoundIsLeafFnCall(Matrix.fromList([
        [90, 1, 0, 0, 0, 0, 1],
      ]), featuresColumnRangesReduced, true);
      mockBoundLeafLabelFactoryFnCall(
        leafObservations: Matrix.fromList([
          [90, 1, 0, 0, 0, 0, 1],
        ]),
        expectedLabel: DecisionTreeLeafLabel
            .nominal(Vector.fromList([0, 0, 1])),
      );

      mockBoundIsLeafFnCall(Matrix.fromList([
        [23, 0, 0, 1, 0, 1, 0],
      ]), featuresColumnRangesReduced, true);
      mockBoundLeafLabelFactoryFnCall(
        leafObservations: Matrix.fromList([
          [23, 0, 0, 1, 0, 1, 0],
        ]),
        expectedLabel: DecisionTreeLeafLabel
            .nominal(Vector.fromList([0, 1, 0])),
      );

      mockBoundIsLeafFnCall(Matrix.fromList([
        [55, 0, 1, 0, 1, 0, 0],
      ]), featuresColumnRangesReduced, true);
      mockBoundLeafLabelFactoryFnCall(
        leafObservations: Matrix.fromList([
          [55, 0, 1, 0, 1, 0, 0],
        ]),
        expectedLabel: DecisionTreeLeafLabel
            .nominal(Vector.fromList([1, 0, 0])),
      );

      final rootNode = DecisionTreeSolver(
        samples,
        featuresColumnRangesFull,
        outcomesColumnRange,
        rangeToNominalValues,
        leafDetector,
        leafLabelFactory,
        bestStumpFinder,
      ).root;

      testTreeNode(rootNode,
        shouldBeLeaf: false,
        expectedSplittingNumericalValue: null,
        expectedSplittingNominalValue: nominalFeatureValues,
        expectedSplittingColumnRange: nominalFeatureRange,
        expectedChildrenLength: 3,
        expectedLabel: null,
      );

      final firstLevelNodes = rootNode.children;
      final secondLevelNodes = firstLevelNodes.first.children;

      testTreeNode(firstLevelNodes[0],
        shouldBeLeaf: false,
        expectedSplittingNumericalValue: 10,
        expectedSplittingNominalValue: null,
        expectedSplittingColumnRange: ZRange.singleton(0),
        expectedChildrenLength: 2,
        expectedLabel: null,
      );

      testTreeNode(firstLevelNodes[1],
        shouldBeLeaf: true,
        expectedSplittingNumericalValue: null,
        expectedSplittingNominalValue: null,
        expectedSplittingColumnRange: null,
        expectedChildrenLength: null,
        expectedLabel: DecisionTreeLeafLabel
            .nominal(Vector.fromList([0, 1, 0])),
      );

      testTreeNode(secondLevelNodes.first,
        shouldBeLeaf: true,
        expectedSplittingNumericalValue: null,
        expectedSplittingNominalValue: null,
        expectedSplittingColumnRange: null,
        expectedChildrenLength: null,
        expectedLabel: DecisionTreeLeafLabel
            .nominal(Vector.fromList([0, 1, 0])),
      );

      testTreeNode(secondLevelNodes.last,
        shouldBeLeaf: true,
        expectedSplittingNumericalValue: null,
        expectedSplittingNominalValue: null,
        expectedSplittingColumnRange: null,
        expectedChildrenLength: null,
        expectedLabel: DecisionTreeLeafLabel
            .nominal(Vector.fromList([0, 0, 1])),
      );

      verify(leafDetector.isLeaf(any, outcomesColumnRange,
          featuresColumnRangesFull)).called(1);
      verify(leafDetector.isLeaf(any, outcomesColumnRange,
          featuresColumnRangesReduced)).called(5);

      verify(bestStumpFinder.select(
          any,
          any,
          argThat(unorderedEquals(featuresColumnRangesFull)),
          any)
      ).called(1);

      verify(bestStumpFinder.select(
          any,
          any,
          argThat(unorderedEquals(featuresColumnRangesReduced)),
          any)
      ).called(1);
    });
  });
}

void mockLeafDetectorCall(
    LeafDetector leafDetector,
    Matrix split,
    ZRange outcomesColumnRange,
    Iterable<ZRange> featureColumnRanges,
    bool returnValue,
) {
  when(leafDetector.isLeaf(
    argThat(equals(split)),
    outcomesColumnRange,
    argThat(unorderedEquals(featureColumnRanges)),
  )).thenReturn(returnValue);
}

void mockLeafLabelFactoryCall({
  DecisionTreeLeafLabelFactory leafLabelFactory,
  Matrix leafObservations,
  ZRange outcomesColumnRange,
  bool isClassLabelNominal,
  DecisionTreeLeafLabel expectedLabel,
}) {
  when(leafLabelFactory.create(
      argThat(equals(leafObservations)),
      outcomesColumnRange,
      isClassLabelNominal
  )).thenReturn(expectedLabel);
}

void mockStumpFinderCall(
    SplitSelector bestStumpFinder,
    Matrix input,
    ZRange outcomesColumnRange,
    Set<ZRange> featuresColumnRange,
    Map<ZRange, List<Vector>> rangeToNominalValues,
    List<Matrix> expectedOutput,
    double expectedSplittingValue,
    List<Vector> expectedSplittingNominalValues,
    ZRange expectedSplittingRange,
) {
  when(bestStumpFinder.select(
    argThat(equals(input)),
    outcomesColumnRange,
    argThat(unorderedEquals(featuresColumnRange)),
    rangeToNominalValues,
  )).thenReturn(
    DecisionTreeStump(
      expectedSplittingValue,
      expectedSplittingNominalValues,
      expectedSplittingRange,
      expectedOutput,
    ),
  );
}
