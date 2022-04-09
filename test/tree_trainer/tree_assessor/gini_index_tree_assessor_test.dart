import 'package:ml_algo/src/common/distribution_calculator/distribution_calculator_impl.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/gini_index_tree_assessor.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/majority_tree_assessor.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:test/test.dart';

void main() {
  group('GiniIndexTreeAssessor', () {
    final distributionCalculator = const DistributionCalculatorImpl();

    test('should return gini index based error on node', () {
      final node = Matrix.fromList([
        [10, 30, 40, 0],
        [20, 30, 10, 1],
        [30, 20, 30, 1],
        [40, 10, 20, 2],
      ]);
      final assessor = GiniIndexTreeAssessor(distributionCalculator);
      final error = assessor.getError(node, 3);

      expect(error,
          1 / 4 * (1 - 1 / 4) + 1 / 2 * (1 - 1 / 2) + 1 / 4 * (1 - 1 / 4));
    });

    test(
        'should return `0` gini index based error on node if the node '
        'has only one class label', () {
      final node = Matrix.fromList([
        [10, 30, 0],
        [14, 20, 0],
      ]);
      final assessor = GiniIndexTreeAssessor(distributionCalculator);
      final error = assessor.getError(node, 2);

      expect(error, 0);
    });

    test(
        'should return correct aggregated gini index based error on decision stump',
        () {
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
      final assessor = GiniIndexTreeAssessor(distributionCalculator);
      final error = assessor.getAggregatedError(stump, 2);

      expect(
          error,
          closeTo(
              (1 / 4 * (1 - 1 / 4) +
                          1 / 2 * (1 - 1 / 2) +
                          1 / 4 * (1 - 1 / 4)) *
                      1 /
                      3 +
                  (3 / 4 * (1 - 3 / 4) + 1 / 4 * (1 - 1 / 4)) * 1 / 3 +
                  (1 / 2 * (1 - 1 / 2) +
                          1 / 4 * (1 - 1 / 4) +
                          1 / 4 * (1 - 1 / 4)) *
                      1 /
                      3,
              1e-5));
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
      final assessor = GiniIndexTreeAssessor(distributionCalculator);
      final error = assessor.getAggregatedError(stump, 2);

      expect(
          error,
          (1 / 4 * (1 - 1 / 4) + 1 / 2 * (1 - 1 / 2) + 1 / 4 * (1 - 1 / 4)) *
                  2 /
                  5 +
              (3 / 5 * (1 - 3 / 5) + 2 / 5 * (1 - 2 / 5)) * 1 / 2 +
              0);
    });

    test(
        'should return gini index based error, that is equal to 0, if all '
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
      final assessor = GiniIndexTreeAssessor(distributionCalculator);
      final error = assessor.getAggregatedError(stump, 2);

      expect(error, 0);
    });

    test(
        'should return gini index based error that is equal to 0, if all '
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
      final assessor = GiniIndexTreeAssessor(distributionCalculator);
      final error = assessor.getAggregatedError(stump, 2);

      expect(error, 0);
    });

    test('should ignore empty split matrices', () {
      final node1 = Matrix.fromList([]);

      final node2 = Matrix.fromList([
        [80, 90, 0],
      ]);

      final node3 = Matrix.fromList([
        [80, 90, 1],
      ]);

      final stump = [node1, node2, node3];

      expect(const MajorityTreeAssessor().getAggregatedError(stump, 2), 0);
    });

    test(
        'should return gini index based error, if some nodes of the stump '
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
      final assessor = GiniIndexTreeAssessor(distributionCalculator);
      final error = assessor.getAggregatedError(stump, 2);

      expect(error, 0.5);
    });
  });
}
