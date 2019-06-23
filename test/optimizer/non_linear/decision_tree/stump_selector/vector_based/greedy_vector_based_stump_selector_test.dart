import 'package:ml_algo/src/optimizer/non_linear/decision_tree/stump_selector/vector_based/greedy_vector_based_stump_selector.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:test/test.dart';
import 'package:xrange/zrange.dart';

void main() {
  group('GreedyVectorBasedStumpSelector', () {
    test('should select stump, splitting (greedy) the observations into parts '
        'by given column range', () {
      final observations = Matrix.fromList([
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
        Vector.fromList([1, 0, 0]),
      ];
      final selector = GreedyVectorBasedStumpSelector();
      final stump = selector.select(
          observations,
          splittingColumnRange,
          splittingValues,
      );
      expect(stump, equals([
        [
          [11, 22, 0, 0, 1, 30],
          [60, 23, 0, 0, 1, 20],
        ],
        [
          [13, 99, 0, 1, 0, 30],
        ],
        [
          [20, 25, 1, 0, 0, 10],
          [17, 66, 1, 0, 0, 70],
        ],
      ]));
    });

    test('should return just one node in stump that is equal to the given '
        'matrix if splitting value collection contains the only value and the '
        'observations contain just this value in the target column range', () {
      final observations = Matrix.fromList([
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
      final selector = GreedyVectorBasedStumpSelector();
      final stump = selector.select(
        observations,
        splittingColumnRange,
        splittingValues,
      );
      expect(stump, equals([
        [
          [11, 22, 0, 0, 1, 30],
          [60, 23, 0, 0, 1, 20],
          [20, 25, 0, 0, 1, 10],
          [17, 66, 0, 0, 1, 70],
          [13, 99, 0, 0, 1, 30],
        ],
      ]));
    });

    test('should return just one node in stump that is just a part of the '
        'given matrix if splitting value collection contains the only value '
        'and the observations contain different values in the target column '
        'range', () {
      final observations = Matrix.fromList([
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
      final selector = GreedyVectorBasedStumpSelector();
      final stump = selector.select(
        observations,
        splittingColumnRange,
        splittingValues,
      );
      expect(stump, equals([
        [
          [11, 22, 0, 0, 1, 30],
          [60, 23, 0, 0, 1, 20],
        ],
      ]));
    });

    test('should return an empty stum if splitting value collection is '
        'empty', () {
      final observations = Matrix.fromList([
        [11, 22, 0, 0, 1, 30],
        [60, 23, 0, 0, 1, 20],
        [20, 25, 1, 0, 0, 10],
        [17, 66, 1, 0, 0, 70],
        [13, 99, 0, 1, 0, 30],
      ]);
      final splittingColumnRange = ZRange.closed(2, 4);
      final splittingValues = <Vector>[];
      final selector = GreedyVectorBasedStumpSelector();
      final stump = selector.select(
        observations,
        splittingColumnRange,
        splittingValues,
      );
      expect(stump, equals(<Matrix>[]));
    });

    test('should return an empty stump if no one value from the splitting'
        'value collection is not contained in the target column range', () {
      final observations = Matrix.fromList([
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
      final selector = GreedyVectorBasedStumpSelector();
      final stump = selector.select(
        observations,
        splittingColumnRange,
        splittingValues,
      );
      expect(stump, equals(<Matrix>[]));
    });

    test('should not throw an error if at least one\'s length of the given '
        'splitting vectors does not match the length of the target column'
        'range', () {
      final observations = Matrix.fromList([
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
      final selector = GreedyVectorBasedStumpSelector();
      final stump = selector.select(
        observations,
        splittingColumnRange,
        splittingValues,
      );

      expect(stump, equals([
        [
          [11, 22, 0, 0, 1, 30],
          [60, 23, 0, 0, 1, 20],
        ],
        [
          [13, 99, 0, 1, 0, 30],
        ],
      ]));
    });

    test('should throw an error if unappropriate range is given (left boundary '
        'is less than 0)', () {
      final observations = Matrix.fromList([
        [11, 22, 0, 0, 1, 30],
        [60, 23, 0, 0, 1, 20],
        [20, 25, 1, 0, 0, 10],
        [17, 66, 1, 0, 0, 70],
        [13, 99, 0, 1, 0, 30],
      ]);
      final splittingColumnRange = ZRange.closed(-2, 4);
      final splittingValues = [
        Vector.fromList([0, 0, 1]),
        Vector.fromList([0, 1, 0]),
      ];
      final selector = GreedyVectorBasedStumpSelector();
      final actual = () => selector.select(
        observations,
        splittingColumnRange,
        splittingValues,
      );
      expect(actual, throwsException);
    });

    test('should throw an error if unappropriate range is given (right boundary '
        'is greater than the observations columns number)', () {
      final observations = Matrix.fromList([
        [11, 22, 0, 0, 1, 30],
        [60, 23, 0, 0, 1, 20],
        [20, 25, 1, 0, 0, 10],
        [17, 66, 1, 0, 0, 70],
        [13, 99, 0, 1, 0, 30],
      ]);
      final splittingColumnRange = ZRange.closed(0, 10);
      final splittingValues = [
        Vector.fromList([0, 0, 1]),
        Vector.fromList([0, 1, 0]),
      ];
      final selector = GreedyVectorBasedStumpSelector();
      final actual = () => selector.select(
        observations,
        splittingColumnRange,
        splittingValues,
      );
      expect(actual, throwsException);
    });
  });
}
