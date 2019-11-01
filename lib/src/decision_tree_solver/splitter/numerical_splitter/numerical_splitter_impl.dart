import 'package:ml_algo/src/decision_tree_solver/decision_tree_node.dart';
import 'package:ml_algo/src/decision_tree_solver/splitter/numerical_splitter/numerical_splitter.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class NumericalDecisionTreeSplitterImpl implements
    NumericalDecisionTreeSplitter {

  const NumericalDecisionTreeSplitterImpl();

  @override
  Map<DecisionTreeNode, Matrix> split(Matrix samples, int splittingIdx,
      double splittingValue) {
    final left = <Vector>[];
    final right = <Vector>[];
    final splittingClause =
        (Vector sample) => sample[splittingIdx] < splittingValue;
    final oppositeClause = (Vector sample) => !splittingClause(sample);

    samples.rows.forEach((row) => splittingClause(row)
        ? left.add(row)
        : right.add(row));

    final createNode = (TestSamplePredicate splittingClause) =>
        DecisionTreeNode(splittingClause, splittingValue, splittingIdx,
            null, null);

    final leftNode = createNode(splittingClause);
    final rightNode = createNode(oppositeClause);

    return {
      leftNode: Matrix.fromRows(left),
      rightNode: Matrix.fromRows(right),
    };
  }
}
