import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_node.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/splitter/numerical_splitter/numerical_splitter.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:xrange/zrange.dart';

class NumericalSplitterImpl implements NumericalSplitter {
  const NumericalSplitterImpl();

  @override
  Map<DecisionTreeNode, Matrix> split(Matrix samples, ZRange splittingRange,
      double splittingValue) {
    final left = <Vector>[];
    final right = <Vector>[];
    final index = splittingRange.firstValue;
    final splittingClause = (Vector sample) => sample[index] < splittingValue;
    final oppositeClause = (Vector sample) => !splittingClause(sample);

    samples.rows.forEach((row) => splittingClause(row)
        ? left.add(row)
        : right.add(row));

    final createNode = (FilterPredicate splittingClause) =>
        DecisionTreeNode(splittingClause, splittingValue, null, splittingRange,
            null, null);

    final leftNode = createNode(splittingClause);
    final rightNode = createNode(oppositeClause);

    return {
      leftNode: Matrix.fromRows(left),
      rightNode: Matrix.fromRows(right),
    };
  }
}
