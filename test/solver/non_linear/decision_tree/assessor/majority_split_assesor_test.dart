import 'package:ml_algo/src/solver/non_linear/decision_tree/split_assessor/majority_split_assessor.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:test/test.dart';
import 'package:xrange/zrange.dart';

void main() {
  group('MajorityStumpAssessor', () {
    group('when vectors are used as class labels', () {
      test('should return majority-based error on node', () {
        final node = Matrix.fromList([
          [10, 30, 40, 1, 0, 0],
          [20, 30, 10, 0, 0, 1],
          [30, 20, 30, 0, 0, 1],
          [40, 10, 20, 0, 1, 0],
        ]);
        final assessor = const MajoritySplitAssessor();
        final outcomesRange = ZRange.closed(3, 5);
        final error = assessor.getError(node, outcomesRange);
        expect(error, 0.5);
      });

      test('should return 0 majority-based error on node if the node has only '
          'one class label', () {
        final node = Matrix.fromList([
          [10, 30, 1, 0, 0],
          [14, 20, 1, 0, 0],
        ]);
        final assessor = const MajoritySplitAssessor();
        final outcomesRange = ZRange.closed(2, 4);
        final error = assessor.getError(node, outcomesRange);
        expect(error, 0);
      });

      test('should return majority-based error on decision stump when all nodes'
          'in the stump have distinct majority class', () {
        final node1 = Matrix.fromList([
          [10, 30, 1, 0, 0],
          [10, 30, 0, 0, 1],
          [10, 30, 0, 0, 1],
          [10, 30, 0, 1, 0],
        ]);

        final node2 = Matrix.fromList([
          [10, 30, 0, 1, 0],
          [10, 30, 0, 1, 0],
          [10, 30, 0, 1, 0],
          [10, 30, 0, 0, 1],
        ]);

        final node3 = Matrix.fromList([
          [10, 30, 1, 0, 0],
          [10, 30, 1, 0, 0],
          [10, 30, 0, 0, 1],
          [10, 30, 0, 1, 0],
        ]);

        final stump = [node1, node2, node3];
        final assessor = const MajoritySplitAssessor();
        final outcomesRange = ZRange.closed(2, 4);
        final error = assessor.getAggregatedError(stump, outcomesRange);

        expect(error, 5 / 12);
      });

      test('should return correct error if nodes have different length', () {
        final node1 = Matrix.fromList([
          [10, 30, 1, 0, 0],
          [10, 30, 0, 0, 1],
          [10, 30, 0, 0, 1],
          [10, 30, 0, 1, 0],
        ]);

        final node2 = Matrix.fromList([
          [10, 30, 0, 1, 0],
          [10, 30, 0, 1, 0],
          [10, 30, 0, 1, 0],
          [10, 30, 0, 0, 1],
          [10, 30, 0, 0, 1],
        ]);

        final node3 = Matrix.fromList([
          [10, 30, 1, 0, 0],
        ]);

        final stump = [node1, node2, node3];
        final assessor = const MajoritySplitAssessor();
        final outcomesRange = ZRange.closed(2, 4);
        final error = assessor.getAggregatedError(stump, outcomesRange);

        expect(error, 4 / 10);
      });

      test('should return majority-based error, that is equal to 0, if all '
          'nodes in the stump have only one class', () {
        final node1 = Matrix.fromList([
          [10, 30, 1, 0, 0],
          [10, 30, 1, 0, 0],
          [10, 30, 1, 0, 0],
          [10, 30, 1, 0, 0],
        ]);

        final node2 = Matrix.fromList([
          [10, 30, 0, 1, 0],
          [10, 30, 0, 1, 0],
          [10, 30, 0, 1, 0],
          [10, 30, 0, 1, 0],
        ]);

        final node3 = Matrix.fromList([
          [10, 30, 0, 0, 1],
          [10, 30, 0, 0, 1],
          [10, 30, 0, 0, 1],
          [10, 30, 0, 0, 1],
        ]);

        final stump = [node1, node2, node3];
        final assessor = const MajoritySplitAssessor();
        final outcomesRange = ZRange.closed(2, 4);
        final error = assessor.getAggregatedError(stump, outcomesRange);

        expect(error, 0);
      });

      test('should return majority-based error, that is equal to 0, if all '
          'nodes in the stump have only one observation', () {
        final node1 = Matrix.fromList([
          [50, 70, 1, 0, 0],
        ]);

        final node2 = Matrix.fromList([
          [50, 70, 0, 1, 0],
        ]);

        final node3 = Matrix.fromList([
          [50, 70, 0, 0, 1],
        ]);

        final stump = [node1, node2, node3];
        final assessor = const MajoritySplitAssessor();
        final outcomesRange = ZRange.closed(2, 4);
        final error = assessor.getAggregatedError(stump, outcomesRange);

        expect(error, 0);
      });

      test('should throw an error if at least one node in the stump does not '
          'have observations at all', () {
        final node1 = Matrix.fromList([]);

        final node2 = Matrix.fromList([
          [80, 90, 0, 1, 0],
        ]);

        final node3 = Matrix.fromList([
          [80, 90, 0, 0, 1],
        ]);

        final stump = [node1, node2, node3];

        expect(
            () => const MajoritySplitAssessor()
                .getAggregatedError(stump, ZRange.closed(2, 4)),
            throwsException,
        );
      });

      test('should return majority-based error, if some nodes of the stump '
          'have equal quantity of class labels', () {
        final node1 = Matrix.fromList([
          [20, 30, 1, 0, 0],
          [20, 30, 1, 0, 0],
          [20, 30, 0, 0, 0],
          [20, 30, 0, 0, 0],
        ]);

        final node2 = Matrix.fromList([
          [20, 30, 0, 1, 0],
          [20, 30, 0, 1, 0],
          [20, 30, 1, 0, 0],
          [20, 30, 1, 0, 0],
        ]);

        final node3 = Matrix.fromList([
          [20, 30, 0, 0, 1],
          [20, 30, 0, 0, 1],
          [20, 30, 0, 1, 0],
          [20, 30, 0, 1, 0],
        ]);

        final stump = [node1, node2, node3];
        final assessor = const MajoritySplitAssessor();
        final outcomesRange = ZRange.closed(2, 4);
        final error = assessor.getAggregatedError(stump, outcomesRange);

        expect(error, 0.5);
      });
    });

    group('when real values are used as class labels', () {
      test('should return majority-based error on decision stump when all nodes'
          'in the stump have distinct majority class', () {
        final outcomes1 = Vector.fromList([0, 1, 1, 0]);
        final outcomes2 = Vector.fromList([2, 2, 2, 3]);
        final outcomes3 = Vector.fromList([3, 1, 1, 3]);
        final stump = [
          Matrix.fromColumns([Vector.randomFilled(4), outcomes1]),
          Matrix.fromColumns([Vector.randomFilled(4), outcomes2]),
          Matrix.fromColumns([Vector.randomFilled(4), outcomes3]),
        ];
        final assessor = const MajoritySplitAssessor();
        final outcomesRange = ZRange.singleton(1);
        final error = assessor.getAggregatedError(stump, outcomesRange);

        expect(error, 5 / 12);
      });

      test('should return correct error if nodes have different length', () {
        final outcomes1 = Vector.fromList([0, 1, 1, 0]);
        final outcomes2 = Vector.fromList([2, 2, 2, 3, 5]);
        final outcomes3 = Vector.fromList([3, 1, 1, 3, 7, 8]);
        final stump = [
          Matrix.fromColumns([Vector.randomFilled(4), outcomes1]),
          Matrix.fromColumns([Vector.randomFilled(5), outcomes2]),
          Matrix.fromColumns([Vector.randomFilled(6), outcomes3]),
        ];
        final assessor = const MajoritySplitAssessor();
        final outcomesRange = ZRange.singleton(1);
        final error = assessor.getAggregatedError(stump, outcomesRange);

        expect(error, 8 / 15);
      });

      test('should return majority-based error, that is equal to 0, if all '
          'nodes in the stump have only one class', () {
        final outcomes1 = Vector.fromList([0, 0, 0, 0]);
        final outcomes2 = Vector.fromList([1, 1, 1, 1]);
        final outcomes3 = Vector.fromList([2, 2, 2, 2]);
        final stump = [
          Matrix.fromColumns([Vector.randomFilled(4), outcomes1]),
          Matrix.fromColumns([Vector.randomFilled(4), outcomes2]),
          Matrix.fromColumns([Vector.randomFilled(4), outcomes3]),
        ];
        final assessor = const MajoritySplitAssessor();
        final outcomesRange = ZRange.singleton(1);
        final error = assessor.getAggregatedError(stump, outcomesRange);

        expect(error, 0);
      });

      test('should return majority-based error, that is equal to 0, if all '
          'nodes in the stump have only one observation', () {
        final outcomes1 = Vector.fromList([0]);
        final outcomes2 = Vector.fromList([1]);
        final outcomes3 = Vector.fromList([2]);
        final stump = [
          Matrix.fromColumns([Vector.randomFilled(1), outcomes1]),
          Matrix.fromColumns([Vector.randomFilled(1), outcomes2]),
          Matrix.fromColumns([Vector.randomFilled(1), outcomes3]),
        ];
        final assessor = const MajoritySplitAssessor();
        final outcomesRange = ZRange.singleton(1);
        final error = assessor.getAggregatedError(stump, outcomesRange);

        expect(error, 0);
      });

      test('should throw an error if at least one node in the stump does not '
          'have observations at all', () {
        final outcomes1 = Vector.fromList([0, 0, 1]);
        final outcomes2 = Vector.fromList([]);
        final outcomes3 = Vector.fromList([2, 2]);
        final stump = [
          Matrix.fromColumns([Vector.randomFilled(3), outcomes1]),
          Matrix.fromColumns([Vector.fromList([]), outcomes2]),
          Matrix.fromColumns([Vector.randomFilled(2), outcomes3]),
        ];
        expect(
            () => const MajoritySplitAssessor()
                .getAggregatedError(stump, ZRange.singleton(1)),
            throwsException,
        );
      });

      test('should return majority-based error, if some nodes of the stump '
          'have equal quantity of class labels', () {
        final outcomes1 = Vector.fromList([0, 0, 1, 1]);
        final outcomes2 = Vector.fromList([0, 2, 2, 0]);
        final outcomes3 = Vector.fromList([1, 3, 1, 3]);
        final stump = [
          Matrix.fromColumns([Vector.randomFilled(4), outcomes1]),
          Matrix.fromColumns([Vector.randomFilled(4), outcomes2]),
          Matrix.fromColumns([Vector.randomFilled(4), outcomes3]),
        ];
        final assessor = const MajoritySplitAssessor();
        final outcomesRange = ZRange.singleton(1);
        final error = assessor.getAggregatedError(stump, outcomesRange);

        expect(error, 0.5);
      });
    });
  });
}
