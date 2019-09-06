import 'package:ml_algo/src/solver/non_linear/decision_tree/decision_tree_leaf_label.dart';
import 'package:ml_algo/src/solver/non_linear/decision_tree/solver_factory/greedy_solver.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/vector.dart';
import 'package:test/test.dart';
import 'package:xrange/zrange.dart';

import 'test_utils.dart';

void main() {
  group('DecisionTreeSolver', () {
    group('for greedy classifier', () {
      final dataFrame = DataFrame.fromSeries([
        Series('col_1', <int>[10, 90, 23, 55]),
        Series('col_2', <int>[20, 51, 40, 10]),
        Series('col_3', <int>[1, 0, 0, 1], isDiscrete: true),
        Series('col_4', <int>[0, 0, 1, 0], isDiscrete: true),
        Series('col_5', <int>[0, 1, 0, 0], isDiscrete: true),
        Series('col_6', <int>[30, 34, 90, 22]),
        Series('col_7', <int>[40, 31, 50, 80]),
        Series('col_8', <int>[0, 0, 1, 2], isDiscrete: true),
      ]);

      final solver = createGreedySolver(dataFrame, 7, null, 0.3, 1, 3);

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
          expectedSplittingColumnIdx: null,
          expectedChildrenLength: 2,
          expectedLabel: null,
        );

        testTreeNode(rootNode.children.first,
          shouldBeLeaf: true,
          expectedSplittingNumericalValue: 15,
          expectedSplittingNominalValue: null,
          expectedSplittingColumnIdx: 1,
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
          expectedSplittingColumnIdx: 1,
          expectedChildrenLength: 3,
        );

        testTreeNode(rootNode.children.last.children[0],
          shouldBeLeaf: true,
          expectedSplittingNumericalValue: null,
          expectedSplittingNominalValue: Vector.fromList([0, 0, 1]),
          expectedSplittingColumnIdx: nominalFeatureRange,
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
          expectedSplittingColumnIdx: nominalFeatureRange,
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
          expectedSplittingColumnIdx: nominalFeatureRange,
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
