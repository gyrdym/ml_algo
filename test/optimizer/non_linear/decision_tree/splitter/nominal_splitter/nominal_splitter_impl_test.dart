import 'package:ml_algo/src/optimizer/non_linear/decision_tree/splitter/nominal_splitter/nominal_splitter_impl.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:test/test.dart';
import 'package:xrange/zrange.dart';

import '../../test_utils.dart';

void main() {
  group('NominalSplitterImpl', () {
    final splitter = const NominalSplitterImpl();

    test('should perform split only by one splitting value, splitting column '
        'contains only this splitting value', () {
      final samples = Matrix.fromList([
        [11, 22, 0, 0, 1, 30],
        [60, 23, 0, 0, 1, 20],
        [20, 25, 0, 0, 1, 10],
        [17, 66, 0, 0, 1, 70],
        [13, 99, 0, 0, 1, 30],
      ]);
      final splittingColumnRange = ZRange.closed(2, 4);
      final splittingValues = [
        Vector.fromList([0, 0, 1]),
      ];

      final split = splitter.split(samples, splittingColumnRange,
          splittingValues);

      expect(split.values, equals([
        [
          [11, 22, 0, 0, 1, 30],
          [60, 23, 0, 0, 1, 20],
          [20, 25, 0, 0, 1, 10],
          [17, 66, 0, 0, 1, 70],
          [13, 99, 0, 0, 1, 30],
        ],
      ]));
      testTreeNode(split.keys.first,
          shouldBeLeaf: true,
          expectedSplittingNumericalValue: null,
          expectedSplittingNominalValue: Vector.fromList([0, 0, 1]),
          expectedSplittingColumnRange: splittingColumnRange,
          expectedChildrenLength: null,
          expectedLabel: null,
      );
    });

    test('should perform split only by one splitting value, splitting column '
        'contains different values', () {
      final samples = Matrix.fromList([
        [11, 22, 0, 0, 1, 30],
        [60, 23, 0, 0, 1, 20],
        [20, 25, 1, 0, 0, 10],
        [17, 66, 1, 0, 0, 70],
        [13, 99, 0, 1, 0, 30],
      ]);
      final splittingColumnRange = ZRange.closed(2, 4);
      final splittingValues = [
        Vector.fromList([0, 0, 1]),
      ];

      final split = splitter.split(samples, splittingColumnRange,
          splittingValues);

      expect(split.values, equals([
        [
          [11, 22, 0, 0, 1, 30],
          [60, 23, 0, 0, 1, 20],
        ],
      ]));
      testTreeNode(split.keys.first,
        shouldBeLeaf: true,
        expectedSplittingNumericalValue: null,
        expectedSplittingNominalValue: Vector.fromList([0, 0, 1]),
        expectedSplittingColumnRange: splittingColumnRange,
        expectedChildrenLength: null,
        expectedLabel: null,
      );
    });

    test('should return an empty list if no one value from the splitting value '
        'collection is contained in the target column range', () {
      final samples = Matrix.fromList([
        [11, 22, 0, 0, 1, 30],
        [60, 23, 0, 0, 1, 20],
        [20, 25, 1, 0, 0, 10],
        [17, 66, 1, 0, 0, 70],
        [13, 99, 0, 1, 0, 30],
      ]);
      final splittingColumnRange = ZRange.closed(2, 4);
      final splittingValues = [
        Vector.randomFilled(3),
        Vector.randomFilled(3),
        Vector.randomFilled(3),
      ];

      final split = splitter.split(samples, splittingColumnRange,
          splittingValues);

      expect(split, hasLength(0));
    });

    test('should ignore splitting vectors with improper length', () {
      final samples = Matrix.fromList([
        [11, 22, 0, 0, 1, 30],
        [60, 23, 0, 0, 1, 20],
        [20, 25, 1, 0, 0, 10],
        [17, 66, 1, 0, 0, 70],
        [13, 99, 0, 1, 0, 30],
      ]);
      final splittingColumnRange = ZRange.closed(2, 4);
      final splittingValues = [
        Vector.fromList([0, 0, 1]),
        Vector.fromList([0, 1, 0]),
        Vector.fromList([1, 0, 0, 0]),
      ];

      final split = splitter.split(samples, splittingColumnRange,
          splittingValues);

      expect(split.values, equals([
        [
          [11, 22, 0, 0, 1, 30],
          [60, 23, 0, 0, 1, 20],
        ],
        [
          [13, 99, 0, 1, 0, 30],
        ],
      ]));

      testTreeNode(split.keys.first,
        shouldBeLeaf: true,
        expectedSplittingNumericalValue: null,
        expectedSplittingNominalValue: Vector.fromList([0, 0, 1]),
        expectedSplittingColumnRange: splittingColumnRange,
        expectedChildrenLength: null,
        expectedLabel: null,
        samplesToCheck: {
          Vector.fromList([1e3, -22, 0, 0, 1, 30000]): true,
          Vector.fromList([1.3, 22, 0, 0, 1, 11111]): true,
          Vector.fromList([1.3, 22, 0, 0, 1]): true,
          Vector.fromList([1e3, -22, 0, 1, 0, 30000]): false,
          Vector.fromList([1e3, -22, 1, 1, 1, 30000]): false,
          Vector.fromList([1e3, 0, 0, 1, 30000]): false,
        }
      );

      testTreeNode(split.keys.last,
        shouldBeLeaf: true,
        expectedSplittingNumericalValue: null,
        expectedSplittingNominalValue: Vector.fromList([0, 1, 0]),
        expectedSplittingColumnRange: splittingColumnRange,
        expectedChildrenLength: null,
        expectedLabel: null,
        samplesToCheck: {
          Vector.fromList([1e3, -22, 0, 1, 0, 30000]): true,
          Vector.fromList([1e3, -22, 0, 1, 0]): true,
          Vector.fromList([0, 0, 0, 1, 0]): true,
          Vector.fromList([1e3, -22, 0, 0, 1, 30000]): false,
          Vector.fromList([1e3, -22, 33, 20, 1, 30000]): false,
          Vector.fromList([1e3, 0, 1, 0, 30000]): false,
        }
      );
    });
  });
}
