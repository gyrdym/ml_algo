import 'package:ml_algo/src/classifier/decision_tree/greedy_classifier_dependencies.dart';
import 'package:ml_algo/src/common/sequence_elements_distribution_calculator/distribution_calculator_impl.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_leaf_label.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_solver.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_detector/leaf_detector.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_detector/leaf_detector_impl.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_label_factory/leaf_label_factory.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_label_factory/majority_leaf_label_factory.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/split_assessor/majority_split_assessor.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/split_selector/greedy_split_selector.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/split_selector/split_selector.dart';
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
    group('for greedy classifier', () {
      test('should build a tree', () {
        final dependencies = getGreedyDecisionTreeDependencies(0.3, 1);

        final observations = Matrix.fromList([
          [10, 20, 1, 0, 0, 30, 40, 0, 0, 1],
          [90, 51, 0, 0, 1, 34, 31, 0, 0, 1],
          [23, 40, 0, 1, 0, 90, 50, 0, 1, 0],
          [55, 10, 1, 0, 0, 22, 80, 1, 0, 0],
        ]);

        final nominalFeatureRange = ZRange.closed(2, 4);
        final outcomesColumnRange = ZRange.closed(7, 9);

        final featuresColumnRanges = Set<ZRange>.from(<ZRange>[
          ZRange.singleton(0),
          ZRange.singleton(1),
          nominalFeatureRange,
          ZRange.singleton(5),
          ZRange.singleton(6),
        ]);

        final rangeToNominalValues = {
          nominalFeatureRange: [
            // order of nominal features is valuable for building the tree -
            // the earlier value is in the list, the earlier the appropriate
            // node will be built
            Vector.fromList([0, 0, 1]),
            Vector.fromList([0, 1, 0]),
            Vector.fromList([1, 0, 0]),
          ],
          outcomesColumnRange: [
            Vector.fromList([0, 0, 1]),
            Vector.fromList([0, 1, 0]),
            Vector.fromList([1, 0, 0]),
          ],
        };

        final rootNode = DecisionTreeSolver(
          observations,
          featuresColumnRanges,
          outcomesColumnRange,
          rangeToNominalValues,
          dependencies.getDependency<LeafDetector>(),
          dependencies.getDependency<DecisionTreeLeafLabelFactory>(),
          dependencies.getDependency<SplitSelector>(),
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
          expectedChildrenLength: 3,
        );

        testTreeNode(rootNode.children.last.children[0],
          shouldBeLeaf: true,
          expectedSplittingNumericalValue: null,
          expectedSplittingNominalValue: Vector.fromList([0, 0, 1]),
          expectedSplittingColumnRange: nominalFeatureRange,
          expectedChildrenLength: null,
          expectedLabel: DecisionTreeLeafLabel.nominal(
            Vector.fromList([0, 0, 1]),
            probability: 1,
          ),
        );

        testTreeNode(rootNode.children.last.children[1],
          shouldBeLeaf: true,
          expectedSplittingNumericalValue: null,
          expectedSplittingNominalValue: Vector.fromList([0, 1, 0]),
          expectedSplittingColumnRange: nominalFeatureRange,
          expectedChildrenLength: null,
          expectedLabel: DecisionTreeLeafLabel.nominal(
            Vector.fromList([0, 1, 0]),
            probability: 1,
          ),
        );

        testTreeNode(rootNode.children.last.children[2],
          shouldBeLeaf: true,
          expectedSplittingNumericalValue: null,
          expectedSplittingNominalValue: Vector.fromList([1, 0, 0]),
          expectedSplittingColumnRange: nominalFeatureRange,
          expectedChildrenLength: null,
          expectedLabel: DecisionTreeLeafLabel.nominal(
            Vector.fromList([0, 0, 1]),
            probability: 1,
          ),
        );
      });
    });
  });
}
