import 'package:ml_algo/src/tree_trainer/splitter/numerical_splitter/numerical_splitter.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/split_predicate/_helper/get_split_predicate_by_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/split_predicate/predicate_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class NumericalTreeSplitterImpl implements NumericalTreeSplitter {
  const NumericalTreeSplitterImpl();

  @override
  Map<TreeNode, Matrix> split(
      Matrix samples, int splittingIdx, double splittingValue) {
    final left = <Vector>[];
    final right = <Vector>[];

    final splittingPredicateType = PredicateType.lessThan;
    final oppositeSplittingPredicateType = PredicateType.greaterThanOrEqualTo;

    final splittingPredicate = getSplitPredicateByType(splittingPredicateType);

    samples.rows.forEach(
      (row) => splittingPredicate(row, splittingIdx, splittingValue)
          ? left.add(row)
          : right.add(row),
    );

    final createNode = (PredicateType predicateType) => TreeNode(
          predicateType,
          splittingValue,
          splittingIdx,
          null,
          null,
        );

    final leftNode = createNode(splittingPredicateType);
    final rightNode = createNode(oppositeSplittingPredicateType);

    return {
      leftNode: Matrix.fromRows(left, dtype: samples.dtype),
      rightNode: Matrix.fromRows(right, dtype: samples.dtype),
    };
  }
}
