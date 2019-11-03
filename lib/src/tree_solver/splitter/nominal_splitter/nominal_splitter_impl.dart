import 'package:ml_algo/src/tree_solver/tree_node.dart';
import 'package:ml_algo/src/tree_solver/splitter/nominal_splitter/nominal_splitter.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class NominalTreeSplitterImpl implements NominalTreeSplitter {
  const NominalTreeSplitterImpl();

  @override
  Map<TreeNode, Matrix> split(Matrix samples, int splittingIdx,
      List<num> uniqueValues) =>
      Map.fromEntries(uniqueValues.map((value) {
        final splittingClause =
            (Vector sample) => sample[splittingIdx] == value;

        final foundRows = samples.rows.where(splittingClause)
            .toList(growable: false);

        final node = TreeNode(splittingClause, value,
            splittingIdx, null, null);

        return MapEntry(node, Matrix.fromRows(foundRows));
      }).where((entry) => entry.value.rowsNum > 0),
  );
}