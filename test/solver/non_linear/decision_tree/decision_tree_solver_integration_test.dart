import 'package:ml_algo/src/solver/non_linear/decision_tree/decision_tree_leaf_label.dart';
import 'package:ml_algo/src/solver/non_linear/decision_tree/solver_factory/greedy_solver.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:test/test.dart';
import 'package:xrange/zrange.dart';

import 'test_utils.dart';

void main() {
  group('DecisionTreeSolver', () {
    group('for greedy classifier', () {
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

      final solver = createGreedySolver(
        observations,
        featuresColumnRanges,
        outcomesColumnRange,
        rangeToNominalValues,
        0.3,
        1,
        3,
      );

      test('should build a tree', () {
        final rootNode = solver.root;

        /*
         *          The tree structure:
         *
         *                 (root)
         *                *     *
         *            *             *
         *         *                    *
         *  index: 1                 index: 1
         *  condition: <15           condition: >=15
         *  prediction: 1,0,0           * * *
         *                          *     *      *
         *                      *         *           *
         *                  *             *                *
         *    index: (2-4)         index: (2-4)            index: (2-4)
         *    condition: == 0,0,1  condition: == 0,1,0     condition: == 1,0,0
         *    prediction: 0,0,1    prediction: 0,1,0       prediction: 0,0,1
         */

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

      test('should get a label for given sample, leaf 1', () {
        final sample = Vector.fromList([40, 10, 1, 0, 1, 0, 10040]);
        final label = solver.getLabelForSample(sample);
        testLeafLabel(label, DecisionTreeLeafLabel.nominal(
          Vector.fromList([1, 0, 0]), probability: 1));
      });

      test('should get a label for given sample, leaf 2', () {
        final sample = Vector.fromList([0, 100, 0, 0, 1, 10040, 300]);
        final label = solver.getLabelForSample(sample);
        testLeafLabel(label, DecisionTreeLeafLabel.nominal(
            Vector.fromList([0, 0, 1]), probability: 1));
      });

      test('should get a label for given sample, leaf 3', () {
        final sample = Vector.fromList([0, 100, 0, 1, 0, 10040, 310]);
        final label = solver.getLabelForSample(sample);
        testLeafLabel(label, DecisionTreeLeafLabel.nominal(
            Vector.fromList([0, 1, 0]), probability: 1));
      });

      test('should get a label for given sample, leaf 4', () {
        final sample = Vector.fromList([1220, 1200, 1, 0, 0, -10, 1530]);
        final label = solver.getLabelForSample(sample);
        testLeafLabel(label, DecisionTreeLeafLabel.nominal(
            Vector.fromList([0, 0, 1]), probability: 1));
      });

      test('should throw an error if nominal value in the given sample does '
          'not belong to nominal ones which were used while building the '
          'tree', () {
        final sample = Vector.fromList([-1e3, 40, 1, 1, 1, -10, 1530]);
        final actual = () => solver.getLabelForSample(sample);

        expect(actual, throwsException);
      });
    });
  });
}
