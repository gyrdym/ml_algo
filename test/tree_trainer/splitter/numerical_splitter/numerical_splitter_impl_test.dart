import 'package:ml_algo/src/tree_trainer/splitter/numerical_splitter/numerical_splitter_impl.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/decision_tree_node/decision_intermediate_tree_node_factory.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/decision_tree_node/decision_tree_node.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:test/test.dart';

import '../../test_utils.dart';

void main() {
  group('NumericalTreeSplitterImpl', () {
    final nodeFactory = const DecisionIntermediateTreeNodeFactory();

    test(
        'should split given matrix into two parts: first part should contain '
        'values less than the splitting value, right part should contain '
        'values greater than or equal to the splitting value', () {
      final samples = Matrix.fromList([
        [111, 2, 30, 4],
        [1, 32, 10, 44],
        [11, 22, 10, 14],
        [33, 12, 5, 55],
        [0, 20, 60, 10],
      ]);
      final splittingIdx = 2;
      final splittingValue = 10.0;
      final splitter = NumericalTreeSplitterImpl(nodeFactory);
      final actual = splitter.split<DecisionTreeNode>(
          samples, splittingIdx, splittingValue);

      expect(
          actual.values,
          equals([
            [
              [33, 12, 5, 55],
            ],
            [
              [111, 2, 30, 4],
              [1, 32, 10, 44],
              [11, 22, 10, 14],
              [0, 20, 60, 10],
            ],
          ]));

      testTreeNode(
        actual.keys.first,
        shouldBeLeaf: true,
        expectedSplittingValue: splittingValue,
        expectedSplittingColumnIdx: splittingIdx,
        expectedChildrenLength: null,
        expectedLabel: null,
      );

      testTreeNode(
        actual.keys.last,
        shouldBeLeaf: true,
        expectedSplittingValue: splittingValue,
        expectedSplittingColumnIdx: splittingIdx,
        expectedChildrenLength: null,
        expectedLabel: null,
      );
    });

    test(
        'should split given matrix into two parts if all the values are '
        'greater than or equal to the splitting value: the first part should be '
        'empty', () {
      final samples = Matrix.fromList([
        [111, 2, 30, 4],
        [1, 32, 10, 44],
        [11, 22, 10, 14],
        [33, 12, 500, 55],
        [0, 20, 60, 10],
      ]);
      final splittingIdx = 2;
      final splittingValue = 10.0;
      final splitter = NumericalTreeSplitterImpl(nodeFactory);
      final actual = splitter.split<DecisionTreeNode>(
          samples, splittingIdx, splittingValue);

      expect(
          actual.values,
          equals([
            <double>[],
            [
              [111, 2, 30, 4],
              [1, 32, 10, 44],
              [11, 22, 10, 14],
              [33, 12, 500, 55],
              [0, 20, 60, 10],
            ],
          ]));

      testTreeNode(
        actual.keys.first,
        shouldBeLeaf: true,
        expectedSplittingValue: splittingValue,
        expectedSplittingColumnIdx: splittingIdx,
        expectedChildrenLength: null,
        expectedLabel: null,
      );

      testTreeNode(
        actual.keys.last,
        shouldBeLeaf: true,
        expectedSplittingValue: splittingValue,
        expectedSplittingColumnIdx: splittingIdx,
        expectedChildrenLength: null,
        expectedLabel: null,
      );
    });

    test(
        'should split given matrix into two parts if splitting value is '
        '0', () {
      final samples = Matrix.fromList([
        [111, 2, 30, 4],
        [1, 32, -10, 44],
        [11, 22, 0, 14],
        [33, 12, 500, 55],
        [0, 20, -60, 10],
      ]);
      final splittingIdx = 2;
      final splittingValue = 0.0;
      final splitter = NumericalTreeSplitterImpl(nodeFactory);
      final actual = splitter.split<DecisionTreeNode>(
          samples, splittingIdx, splittingValue);

      expect(
          actual.values,
          equals([
            [
              [1, 32, -10, 44],
              [0, 20, -60, 10],
            ],
            [
              [111, 2, 30, 4],
              [11, 22, 0, 14],
              [33, 12, 500, 55],
            ],
          ]));

      testTreeNode(
        actual.keys.first,
        shouldBeLeaf: true,
        expectedSplittingValue: splittingValue,
        expectedSplittingColumnIdx: splittingIdx,
        expectedChildrenLength: null,
        expectedLabel: null,
      );

      testTreeNode(
        actual.keys.last,
        shouldBeLeaf: true,
        expectedSplittingValue: splittingValue,
        expectedSplittingColumnIdx: splittingIdx,
        expectedChildrenLength: null,
        expectedLabel: null,
      );
    });

    test(
        'should split given matrix into two parts if all the values are '
        'less than the splitting value: the second part should be empty', () {
      final samples = Matrix.fromList([
        [111, 2, 30, 4],
        [1, 32, 10, 44],
        [11, 22, 10, 14],
        [33, 12, 500, 55],
        [0, 20, 60, 10],
      ]);
      final splittingIdx = 2;
      final splittingValue = 1000.0;
      final splitter = NumericalTreeSplitterImpl(nodeFactory);
      final actual = splitter.split<DecisionTreeNode>(
          samples, splittingIdx, splittingValue);

      expect(
          actual.values,
          equals([
            [
              [111, 2, 30, 4],
              [1, 32, 10, 44],
              [11, 22, 10, 14],
              [33, 12, 500, 55],
              [0, 20, 60, 10],
            ],
            <double>[],
          ]));

      testTreeNode(
        actual.keys.first,
        shouldBeLeaf: true,
        expectedSplittingValue: splittingValue,
        expectedSplittingColumnIdx: splittingIdx,
        expectedChildrenLength: null,
        expectedLabel: null,
      );

      testTreeNode(
        actual.keys.last,
        shouldBeLeaf: true,
        expectedSplittingValue: splittingValue,
        expectedSplittingColumnIdx: splittingIdx,
        expectedChildrenLength: null,
        expectedLabel: null,
      );
    });

    test(
        'should split given matrix into two parts if splitting column index is'
        ' 0', () {
      final samples = Matrix.fromList([
        [111, 2, 30, 4],
        [1, 32, 10, 44],
        [11, 22, 10, 14],
        [33, 12, 500, 55],
        [0, 20, 60, 10],
      ]);

      final splittingIdx = 0;
      final splittingValue = 2.0;
      final splitter = NumericalTreeSplitterImpl(nodeFactory);
      final actual = splitter.split<DecisionTreeNode>(
          samples, splittingIdx, splittingValue);

      expect(
          actual.values,
          equals([
            [
              [1, 32, 10, 44],
              [0, 20, 60, 10],
            ],
            [
              [111, 2, 30, 4],
              [11, 22, 10, 14],
              [33, 12, 500, 55],
            ],
          ]));

      testTreeNode(
        actual.keys.first,
        shouldBeLeaf: true,
        expectedSplittingValue: splittingValue,
        expectedSplittingColumnIdx: splittingIdx,
        expectedChildrenLength: null,
        expectedLabel: null,
      );

      testTreeNode(
        actual.keys.last,
        shouldBeLeaf: true,
        expectedSplittingValue: splittingValue,
        expectedSplittingColumnIdx: splittingIdx,
        expectedChildrenLength: null,
        expectedLabel: null,
      );
    });

    test(
        'should split given matrix into two parts if splitting column is the '
        'last column', () {
      final samples = Matrix.fromList([
        [111, 2, 30, 4],
        [1, 32, 10, 44],
        [11, 22, 10, 14],
        [33, 12, 500, 55],
        [0, 20, 60, 10],
      ]);

      final splittingIdx = 3;
      final splittingValue = 20.0;
      final splitter = NumericalTreeSplitterImpl(nodeFactory);
      final actual = splitter.split<DecisionTreeNode>(
          samples, splittingIdx, splittingValue);

      final leftNodeData = actual.values.first;
      final rightNodeData = actual.values.last;

      final leftNode = actual.keys.first;
      final rightNode = actual.keys.last;

      expect(
          leftNodeData,
          equals([
            [111, 2, 30, 4],
            [11, 22, 10, 14],
            [0, 20, 60, 10],
          ]));

      expect(
          rightNodeData,
          equals([
            [1, 32, 10, 44],
            [33, 12, 500, 55],
          ]));

      testTreeNode(
        leftNode,
        shouldBeLeaf: true,
        expectedSplittingValue: splittingValue,
        expectedSplittingColumnIdx: splittingIdx,
        expectedChildrenLength: null,
        expectedLabel: null,
        samplesToCheck: {
          Vector.fromList([111, 2, 30, 4]): true,
          Vector.fromList([111, 2, 30, -14]): true,
          Vector.fromList([111, 2, 33, 19]): true,
          Vector.fromList([111, 2, 33, 19, 10]): true,
          Vector.fromList([111, 2, 30, 40]): false,
          Vector.fromList([111, 2, 30, 140]): false,
          Vector.fromList([111, 2, 33, 20]): false,
          Vector.fromList([111, 2, 33, 21, 10]): false,
        },
      );

      testTreeNode(
        rightNode,
        shouldBeLeaf: true,
        expectedSplittingValue: splittingValue,
        expectedSplittingColumnIdx: splittingIdx,
        expectedChildrenLength: null,
        expectedLabel: null,
        samplesToCheck: {
          Vector.fromList([111, 2, 30, 4]): false,
          Vector.fromList([111, 2, 30, -14]): false,
          Vector.fromList([111, 2, 33, 19]): false,
          Vector.fromList([111, 2, 33, 19, 10]): false,
          Vector.fromList([111, 2, 30, 40]): true,
          Vector.fromList([111, 2, 30, 140]): true,
          Vector.fromList([111, 2, 33, 20]): true,
          Vector.fromList([111, 2, 33, 21, 10]): true,
        },
      );
    });
  });
}
