import 'package:ml_algo/src/optimizer/non_linear/decision_tree/stump_selector/vector_based/vector_based_stump_selector.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:xrange/zrange.dart';

class GreedyVectorBasedStumpSelector implements VectorBasedStumpSelector {
  @override
  List<Matrix> select(Matrix observations, ZRange splittingColumnRange,
      List<Vector> splittingValues) {
    if (splittingColumnRange.firstValue < 0 ||
        splittingColumnRange.lastValue > observations.columnsNum) {
      throw Exception('Unappropriate range given: $splittingColumnRange, '
          'expected a range within or equal '
          '${ZRange.closed(0, observations.columnsNum)}');
    }
    return splittingValues.map((value) {
      final foundRows = observations.rows
          .where((row) => row.subvectorByRange(splittingColumnRange) == value)
          .toList(growable: false);
      return Matrix.fromRows(foundRows);
    }).where((node) => node.rowsNum > 0).toList(growable: false);
  }
}
