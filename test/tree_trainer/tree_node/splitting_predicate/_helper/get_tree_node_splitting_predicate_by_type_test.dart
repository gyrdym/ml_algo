import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/_helper/get_tree_node_splitting_predicate_by_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/tree_node_splitting_predicate_type.dart';
import 'package:ml_linalg/vector.dart';
import 'package:test/test.dart';

void main() {
  group('getTreeNodeSplittingPredicateByType', () {
    final sample = Vector.fromList([10, 20, 30, 100, 600]);
    final splittingIndex = 2;

    test('should return a less than clause', () {
      final clause = getTreeNodeSplittingPredicateByType(
          TreeNodeSplittingPredicateType.lessThan);
      expect(clause(sample, splittingIndex, 29), isFalse);
      expect(clause(sample, splittingIndex, 30), isFalse);
      expect(clause(sample, splittingIndex, 31), isTrue);
    });

    test('should return a less than or equal to clause', () {
      final clause = getTreeNodeSplittingPredicateByType(
          TreeNodeSplittingPredicateType.lessThanOrEqualTo);
      expect(clause(sample, splittingIndex, 29), isFalse);
      expect(clause(sample, splittingIndex, 30), isTrue);
      expect(clause(sample, splittingIndex, 31), isTrue);
    });

    test('should return an equal to clause', () {
      final clause = getTreeNodeSplittingPredicateByType(
          TreeNodeSplittingPredicateType.equalTo);
      expect(clause(sample, splittingIndex, 29), isFalse);
      expect(clause(sample, splittingIndex, 30), isTrue);
      expect(clause(sample, splittingIndex, 31), isFalse);
    });

    test('should return a greater than or equal to clause', () {
      final clause = getTreeNodeSplittingPredicateByType(
          TreeNodeSplittingPredicateType.greaterThanOrEqualTo);
      expect(clause(sample, splittingIndex, 29), isTrue);
      expect(clause(sample, splittingIndex, 30), isTrue);
      expect(clause(sample, splittingIndex, 31), isFalse);
    });

    test('should return a greater than clause', () {
      final clause = getTreeNodeSplittingPredicateByType(
          TreeNodeSplittingPredicateType.greaterThan);
      expect(clause(sample, splittingIndex, 29), isTrue);
      expect(clause(sample, splittingIndex, 30), isFalse);
      expect(clause(sample, splittingIndex, 31), isFalse);
    });

    test('should throw an error if null is passed as the type', () {
      expect(() => getTreeNodeSplittingPredicateByType(null),
          throwsUnsupportedError);
    });
  });
}
