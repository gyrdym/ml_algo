import 'package:ml_algo/src/common/sequence_elements_distribution_calculator/distribution_calculator_impl.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_leaf_label.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_solver.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_detector/leaf_detector_impl.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_label_factory/majority_leaf_label_factory.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/split_assessor/majority_split_assessor.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/split_selector/greedy_split_selector.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/splitter/greedy_splitter.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/splitter/nominal_splitter/nominal_splitter_impl.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/splitter/numerical_splitter/numerical_splitter_impl.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:test/test.dart';
import 'package:xrange/zrange.dart';

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

      final assessor = const MajoritySplitAssessor();
      final distributionCalculator =
        const SequenceElementsDistributionCalculatorImpl();

      final numericalSplitter = const NumericalSplitterImpl();
      final nominalSplitter = const NominalSplitterImpl();
      final splitter = GreedySplitter(assessor, numericalSplitter,
          nominalSplitter);

      final rootNode = DecisionTreeSolver(
          observations,
          featuresColumnRanges,
          outcomesColumnRange,
          rangeToNominalValues,
          LeafDetectorImpl(assessor, 0.3, 1),
          MajorityLeafLabelFactory(distributionCalculator),
          GreedySplitSelector(assessor, splitter),
      ).root;

      testTreeNode(rootNode,
        shouldBeLeaf: false,
        expectedSplittingNumericalValue: null,
        expectedSplittingNominalValue: null,
        expectedSplittingColumnRange: null,
        expectedChildrenLength: 2,
        expectedLabel: null,
      );

      testTreeNode(rootNode.children.first,
        shouldBeLeaf: true,
        expectedSplittingNumericalValue: 15,
        expectedSplittingNominalValue: null,
        expectedSplittingColumnRange: ZRange.singleton(1),
        expectedChildrenLength: null,
        expectedLabel: DecisionTreeLeafLabel.nominal(
          Vector.fromList([1, 0, 0]),
          probability: 1.0,
        ),
      );

      testTreeNode(rootNode.children.last,
        shouldBeLeaf: false,
        expectedSplittingNumericalValue: 15,
        expectedSplittingNominalValue: null,
        expectedSplittingColumnRange: ZRange.singleton(1),
        expectedChildrenLength: 2,
      );
    });
  });
}
