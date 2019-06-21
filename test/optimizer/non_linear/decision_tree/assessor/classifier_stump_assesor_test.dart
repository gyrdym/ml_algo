import 'package:ml_algo/src/optimizer/non_linear/decision_tree/assessor/classifier_stump_assessor.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:test/test.dart';

void main() {
  group('ClassifierStumpAssessor', () {
    group('when vectors are used as class labels', () {
      test('should return majority-based error on node', () {
        final node = Matrix.fromList([
          [1, 0, 0],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
        ]);
        final error = ClassifierStumpAssessor().getErrorOnNode(node);
        expect(error, 0.5);
      });

      test('should return 0 majority-based error on node if the node has only '
          'one class label', () {
        final node = Matrix.fromList([
          [1, 0, 0],
          [1, 0, 0],
        ]);
        final error = ClassifierStumpAssessor().getErrorOnNode(node);
        expect(error, 0);
      });

      test('should return majority-based error on decision stump when all nodes'
          'in the stump have distinct majority class', () {
        final node1 = Matrix.fromList([
          [1, 0, 0],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
        ]);

        final node2 = Matrix.fromList([
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [0, 0, 1],
        ]);

        final node3 = Matrix.fromList([
          [1, 0, 0],
          [1, 0, 0],
          [0, 0, 1],
          [0, 1, 0],
        ]);

        final stump = [node1, node2, node3];
        final error = ClassifierStumpAssessor().getErrorOnStump(stump);

        expect(error, 1.25);
      });

      test('should return correct error if nodes have different length', () {
        final node1 = Matrix.fromList([
          [1, 0, 0],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
        ]);

        final node2 = Matrix.fromList([
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [0, 0, 1],
          [0, 0, 1],
        ]);

        final node3 = Matrix.fromList([
          [1, 0, 0],
        ]);

        final stump = [node1, node2, node3];
        final error = ClassifierStumpAssessor().getErrorOnStump(stump);

        expect(error, 0.9);
      });

      test('should return majority-based error, that is equal to 0, if all '
          'nodes in the stump have only one class', () {
        final node1 = Matrix.fromList([
          [1, 0, 0],
          [1, 0, 0],
          [1, 0, 0],
          [1, 0, 0],
        ]);

        final node2 = Matrix.fromList([
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
        ]);

        final node3 = Matrix.fromList([
          [0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
        ]);

        final stump = [node1, node2, node3];
        final error = ClassifierStumpAssessor().getErrorOnStump(stump);

        expect(error, 0);
      });

      test('should return majority-based error, that is equal to 0, if all '
          'nodes in the stump have only one observation', () {
        final node1 = Matrix.fromList([
          [1, 0, 0],
        ]);

        final node2 = Matrix.fromList([
          [0, 1, 0],
        ]);

        final node3 = Matrix.fromList([
          [0, 0, 1],
        ]);

        final stump = [node1, node2, node3];
        final error = ClassifierStumpAssessor().getErrorOnStump(stump);

        expect(error, 0);
      });

      test('should throw an error if at least one node in the stump does not '
          'have observations at all', () {
        final node1 = Matrix.fromList([]);

        final node2 = Matrix.fromList([
          [0, 1, 0],
        ]);

        final node3 = Matrix.fromList([
          [0, 0, 1],
        ]);

        final stump = [node1, node2, node3];

        expect(
            () => ClassifierStumpAssessor().getErrorOnStump(stump),
            throwsException,
        );
      });

      test('should return majority-based error, if some nodes of the stump '
          'have equal quantity of class labels', () {
        final node1 = Matrix.fromList([
          [1, 0, 0],
          [1, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
        ]);

        final node2 = Matrix.fromList([
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0],
        ]);

        final node3 = Matrix.fromList([
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
        ]);

        final stump = [node1, node2, node3];
        final error = ClassifierStumpAssessor().getErrorOnStump(stump);

        expect(error, 1.5);
      });
    });

    group('when real values are used as class labels', () {
      test('should return majority-based error on decision stump when all nodes'
          'in the stump have distinct majority class', () {
        final node1 = Vector.fromList([0, 1, 1, 0]);
        final node2 = Vector.fromList([2, 2, 2, 3]);
        final node3 = Vector.fromList([3, 1, 1, 3]);
        final stump = [
          Matrix.fromColumns([node1]),
          Matrix.fromColumns([node2]),
          Matrix.fromColumns([node3]),
        ];
        final error = ClassifierStumpAssessor().getErrorOnStump(stump);
        expect(error, 1.25);
      });

      test('should return correct error if nodes have different length', () {
        final node1 = Vector.fromList([0, 1, 1, 0]);
        final node2 = Vector.fromList([2, 2, 2, 3, 5]);
        final node3 = Vector.fromList([3, 1, 1, 3, 7, 8]);
        final stump = [
          Matrix.fromColumns([node1]),
          Matrix.fromColumns([node2]),
          Matrix.fromColumns([node3]),
        ];
        final error = ClassifierStumpAssessor().getErrorOnStump(stump);
        expect(error, 0.9 + 2/3);
      });

      test('should return majority-based error, that is equal to 0, if all '
          'nodes in the stump have only one class', () {
        final node1 = Vector.fromList([0, 0, 0, 0]);
        final node2 = Vector.fromList([1, 1, 1, 1]);
        final node3 = Vector.fromList([2, 2, 2, 2]);
        final stump = [
          Matrix.fromColumns([node1]),
          Matrix.fromColumns([node2]),
          Matrix.fromColumns([node3]),
        ];
        final error = ClassifierStumpAssessor().getErrorOnStump(stump);
        expect(error, 0);
      });

      test('should return majority-based error, that is equal to 0, if all '
          'nodes in the stump have only one observation', () {
        final node1 = Vector.fromList([0]);
        final node2 = Vector.fromList([1]);
        final node3 = Vector.fromList([2]);
        final stump = [
          Matrix.fromColumns([node1]),
          Matrix.fromColumns([node2]),
          Matrix.fromColumns([node3]),
        ];
        final error = ClassifierStumpAssessor().getErrorOnStump(stump);
        expect(error, 0);
      });

      test('should throw an error if at least one node in the stump does not '
          'have observations at all', () {
        final node1 = Vector.fromList([0, 0, 1]);
        final node2 = Vector.fromList([]);
        final node3 = Vector.fromList([2, 2]);
        final stump = [
          Matrix.fromColumns([node1]),
          Matrix.fromColumns([node2]),
          Matrix.fromColumns([node3]),
        ];
        expect(
            () => ClassifierStumpAssessor().getErrorOnStump(stump),
            throwsException,
        );
      });

      test('should return majority-based error, if some nodes of the stump '
          'have equal quantity of class labels', () {
        final node1 = Vector.fromList([0, 0, 1, 1]);
        final node2 = Vector.fromList([0, 2, 2, 0]);
        final node3 = Vector.fromList([1, 3, 1, 3]);
        final stump = [
          Matrix.fromColumns([node1]),
          Matrix.fromColumns([node2]),
          Matrix.fromColumns([node3]),
        ];
        final error = ClassifierStumpAssessor().getErrorOnStump(stump);
        expect(error, 1.5);
      });
    });
  });
}
