import 'package:ml_algo/src/decision_tree_solver/splitter/nominal_splitter/nominal_splitter_impl.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:test/test.dart';

import '../../test_utils.dart';

void main() {
  group('NominalDecisionTreeSplitterImpl', () {
    final splitter = const NominalDecisionTreeSplitterImpl();

    test('should perform split, splitting column contains only splitting '
        'value', () {
      final samples = Matrix.fromList([
        [11, 22, 1, 30],
        [60, 23, 1, 20],
        [20, 25, 1, 10],
        [17, 66, 1, 70],
        [13, 99, 1, 30],
      ]);
      final splittingColumnIdx = 2;
      final splittingValues = [1.0];

      final split = splitter.split(samples, splittingColumnIdx,
          splittingValues);

      expect(split.values, equals([
        [
          [11, 22, 1, 30],
          [60, 23, 1, 20],
          [20, 25, 1, 10],
          [17, 66, 1, 70],
          [13, 99, 1, 30],
        ],
      ]));
      testTreeNode(split.keys.first,
          shouldBeLeaf: true,
          expectedSplittingValue: 1,
          expectedSplittingColumnIdx: splittingColumnIdx,
          expectedChildrenLength: null,
          expectedLabel: null,
      );
    });

    test('should perform split, splitting column contains different '
        'values', () {
      final samples = Matrix.fromList([
        [11, 22, 1, 30],
        [60, 23, 1, 20],
        [20, 25, 2, 10],
        [17, 66, 2, 70],
        [13, 99, 3, 30],
      ]);
      final splittingColumnIdx = 2;
      final splittingValues = [1.0];

      final split = splitter.split(samples, splittingColumnIdx,
          splittingValues);

      expect(split.values, equals([
        [
          [11, 22, 1, 30],
          [60, 23, 1, 20],
        ],
      ]));
      testTreeNode(split.keys.first,
        shouldBeLeaf: true,
        expectedSplittingValue: 1,
        expectedSplittingColumnIdx: splittingColumnIdx,
        expectedChildrenLength: null,
        expectedLabel: null,
      );
    });

    test('should return an empty list if no one value from the splitting value '
        'collection is contained in the target column range', () {
      final samples = Matrix.fromList([
        [11, 22, 1, 30],
        [60, 23, 1, 20],
        [20, 25, 2, 10],
        [17, 66, 2, 70],
        [13, 99, 3, 30],
      ]);
      final splittingColumnIdx = 2;
      final splittingValues = [
        100.0,
        200.0,
        300.0,
      ];

      final split = splitter.split(samples, splittingColumnIdx,
          splittingValues);

      expect(split, hasLength(0));
    });

    test('should ignore unknown splitting values', () {
      final samples = Matrix.fromList([
        [11, 22, 1, 30],
        [60, 23, 1, 20],
        [20, 25, 2, 10],
        [17, 66, 2, 70],
        [13, 99, 3, 30],
      ]);
      final splittingColumnIdx = 2;
      final splittingValues = [1.0, 3.0, 1000.0];

      final split = splitter.split(samples, splittingColumnIdx,
          splittingValues);

      expect(split.values, equals([
        [
          [11, 22, 1, 30],
          [60, 23, 1, 20],
        ],
        [
          [13, 99, 3, 30],
        ],
      ]));

      testTreeNode(split.keys.first,
        shouldBeLeaf: true,
        expectedSplittingValue: 1,
        expectedSplittingColumnIdx: splittingColumnIdx,
        expectedChildrenLength: null,
        expectedLabel: null,
        samplesToCheck: {
          Vector.fromList([1e3, -22, 1, 30000]): true,
          Vector.fromList([1.3, 22, 1, 11111]): true,
          Vector.fromList([1.3, 22, 1]): true,
          Vector.fromList([1e3, -22, 3, 30000]): false,
          Vector.fromList([1e3, -22, 4, 30000]): false,
          Vector.fromList([1e3, 3, 30000]): false,
        }
      );

      testTreeNode(split.keys.last,
        shouldBeLeaf: true,
        expectedSplittingValue: 3,
        expectedSplittingColumnIdx: splittingColumnIdx,
        expectedChildrenLength: null,
        expectedLabel: null,
        samplesToCheck: {
          Vector.fromList([1e3, -22, 3, 30000]): true,
          Vector.fromList([1e3, -22, 3]): true,
          Vector.fromList([0, 0, 3]): true,
          Vector.fromList([1e3, -22, 2, 30000]): false,
          Vector.fromList([1e3, -22, 4, 30000]): false,
          Vector.fromList([1e3, 3, 30000]): false,
        }
      );
    });
  });
}
