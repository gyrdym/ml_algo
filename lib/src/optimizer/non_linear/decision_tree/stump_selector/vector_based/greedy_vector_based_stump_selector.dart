import 'package:ml_algo/src/optimizer/non_linear/decision_tree/stump_selector/vector_based/vector_based_stump_selector.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:xrange/zrange.dart';

class GreedyVectorBasedStumpSelector implements VectorBasedStumpSelector {
  @override
  List<Matrix> select(Matrix observations, ZRange range,
      List<Vector> splittingValues) => splittingValues.map((value) {
        final foundRows = observations.rows
            .where((row) => row.subvectorByRange(range) == value)
            .toList(growable: false);
        return Matrix.fromRows(foundRows);
      }).toList(growable: false);
}
