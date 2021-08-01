import 'package:ml_algo/src/tree_trainer/split_assessor/majority_split_assessor.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:test/test.dart';

void main() {
  group('MajorityTreeSplitAssessor', () {
    test('should return majority-based error on node', () {
      final node = Matrix.fromList([
        [10, 30, 40, 0],
        [20, 30, 10, 1],
        [30, 20, 30, 1],
        [40, 10, 20, 2],
      ]);
      final assessor = const MajorityTreeSplitAssessor();
      final error = assessor.getError(node, 3);

      expect(error, 0.5);
    });

    test(
        'should return `0` majority-based error on node if the node '
        'has only one class label', () {
      final node = Matrix.fromList([
        [10, 30, 0],
        [14, 20, 0],
      ]);
      final assessor = const MajorityTreeSplitAssessor();
      final error = assessor.getError(node, 2);

      expect(error, 0);
    });

    test(
        'should return majority-based error on decision stump when all nodes'
        'in the stump have distinct majority class', () {
      final node1 = Matrix.fromList([
        [10, 30, 0],
        [10, 30, 1],
        [10, 30, 1],
        [10, 30, 2],
      ]);

      final node2 = Matrix.fromList([
        [10, 30, 2],
        [10, 30, 2],
        [10, 30, 2],
        [10, 30, 1],
      ]);

      final node3 = Matrix.fromList([
        [10, 30, 0],
        [10, 30, 0],
        [10, 30, 1],
        [10, 30, 2],
      ]);

      final stump = [node1, node2, node3];
      final assessor = const MajorityTreeSplitAssessor();
      final error = assessor.getAggregatedError(stump, 2);

      expect(error, 5 / 12);
    });

    test('should return correct error if nodes have different length', () {
      final node1 = Matrix.fromList([
        [10, 30, 0],
        [10, 30, 1],
        [10, 30, 1],
        [10, 30, 2],
      ]);

      final node2 = Matrix.fromList([
        [10, 30, 2],
        [10, 30, 2],
        [10, 30, 2],
        [10, 30, 1],
        [10, 30, 1],
      ]);

      final node3 = Matrix.fromList([
        [10, 30, 0],
      ]);

      final stump = [node1, node2, node3];
      final assessor = const MajorityTreeSplitAssessor();
      final error = assessor.getAggregatedError(stump, 2);

      expect(error, 4 / 10);
    });

    test(
        'should return majority-based error, that is equal to 0, if all '
        'nodes in the stump have only one class', () {
      final node1 = Matrix.fromList([
        [10, 30, 0],
        [10, 30, 0],
        [10, 30, 0],
        [10, 30, 0],
      ]);

      final node2 = Matrix.fromList([
        [10, 30, 1],
        [10, 30, 1],
        [10, 30, 1],
        [10, 30, 1],
      ]);

      final node3 = Matrix.fromList([
        [10, 30, 2],
        [10, 30, 2],
        [10, 30, 2],
        [10, 30, 2],
      ]);

      final stump = [node1, node2, node3];
      final assessor = const MajorityTreeSplitAssessor();
      final error = assessor.getAggregatedError(stump, 2);

      expect(error, 0);
    });

    test(
        'should return majority-based error, that is equal to 0, if all '
        'nodes in the stump have only one observation', () {
      final node1 = Matrix.fromList([
        [50, 70, 0],
      ]);

      final node2 = Matrix.fromList([
        [50, 70, 1],
      ]);

      final node3 = Matrix.fromList([
        [50, 70, 2],
      ]);

      final stump = [node1, node2, node3];
      final assessor = const MajorityTreeSplitAssessor();
      final error = assessor.getAggregatedError(stump, 2);

      expect(error, 0);
    });

    test(
        'should throw an error if at least one node in the stump does not '
        'have observations at all', () {
      final node1 = Matrix.fromList([]);

      final node2 = Matrix.fromList([
        [80, 90, 0],
      ]);

      final node3 = Matrix.fromList([
        [80, 90, 1],
      ]);

      final stump = [node1, node2, node3];

      expect(
        () => const MajorityTreeSplitAssessor().getAggregatedError(stump, 2),
        throwsException,
      );
    });

    test(
        'should return majority-based error, if some nodes of the stump '
        'have equal quantity of class labels', () {
      final node1 = Matrix.fromList([
        [20, 30, 0],
        [20, 30, 0],
        [20, 30, 1],
        [20, 30, 1],
      ]);

      final node2 = Matrix.fromList([
        [20, 30, 2],
        [20, 30, 2],
        [20, 30, 0],
        [20, 30, 0],
      ]);

      final node3 = Matrix.fromList([
        [20, 30, 3],
        [20, 30, 3],
        [20, 30, 2],
        [20, 30, 2],
      ]);

      final stump = [node1, node2, node3];
      final assessor = const MajorityTreeSplitAssessor();
      final error = assessor.getAggregatedError(stump, 2);

      expect(error, 0.5);
    });
  });
}
