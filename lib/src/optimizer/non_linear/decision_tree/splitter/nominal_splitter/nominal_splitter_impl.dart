import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_node.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/splitter/nominal_splitter/nominal_splitter.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:xrange/zrange.dart';

class NominalSplitterImpl implements NominalSplitter {
  const NominalSplitterImpl();

  @override
  Map<DecisionTreeNode, Matrix> split(Matrix samples, ZRange splittingRange,
      List<Vector> nominalValues) => Map.fromEntries(nominalValues.map((value) {
        final splittingClause =
            (Vector sample) => sample.subvectorByRange(splittingRange) == value;

        final foundRows = samples.rows
            .where((row) => row.subvectorByRange(splittingRange) == value)
            .toList(growable: false);

        final node = DecisionTreeNode(splittingClause, null, value,
            splittingRange, null, null);

        return MapEntry(node, Matrix.fromRows(foundRows));
      }).where((entry) => entry.value.rowsNum > 0),
  );
}