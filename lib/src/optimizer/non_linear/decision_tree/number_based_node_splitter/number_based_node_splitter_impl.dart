import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

import 'number_based_node_splitter.dart';

class NumberBasedNodeSplitterImpl implements NumberBasedNodeSplitter {
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
