import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

import 'node_splitter.dart';

class NodeSplitterImpl implements NodeSplitter {
  const NodeSplitterImpl();

  @override
  List<Matrix> split(Matrix observations, int splittingColumnIdx,
      double splittingValue) {
    final source1 = <Vector>[];
    final source2 = <Vector>[];
    observations.rows.forEach((row) => row[splittingColumnIdx] >= splittingValue
        ? source2.add(row)
        : source1.add(row));
    return [Matrix.fromRows(source1), Matrix.fromRows(source2)];
  }
}
