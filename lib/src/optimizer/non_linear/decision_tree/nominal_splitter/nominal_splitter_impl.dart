import 'package:ml_algo/src/optimizer/non_linear/decision_tree/nominal_splitter/nominal_splitter.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:quiver/iterables.dart';
import 'package:xrange/zrange.dart';

class NominalSplitterImpl implements NominalSplitter {
  const NominalSplitterImpl();

  @override
  List<Matrix> split(Matrix samples, ZRange splittingColumnRange,
      List<Vector> nominalValues) =>
      nominalValues.map((value) {
        final foundRows = samples.rows
            .where((row) => row.subvectorByRange(splittingColumnRange) == value)
            .toList(growable: false);
        return Matrix.fromRows(foundRows);
      })
          .where((node) => node.rowsNum > 0)
          .toList(growable: false);

  @override
  Iterable<List<int>> getSplittingIndices(Matrix samples,
      ZRange splittingColumnRange, List<Vector> nominalValues) =>
      nominalValues.map((value) =>
          enumerate(samples.rows)
              .where((indexed) => indexed.value
                .subvectorByRange(splittingColumnRange) == value)
              .map((indexed) => indexed.index)
              .toList(growable: false)
      ).where((indices) => indices.isNotEmpty);
}