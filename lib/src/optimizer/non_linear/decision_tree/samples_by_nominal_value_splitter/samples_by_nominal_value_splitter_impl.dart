import 'package:ml_algo/src/optimizer/non_linear/decision_tree/samples_by_nominal_value_splitter/samples_by_nominal_value_splitter.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:xrange/zrange.dart';

class SamplesByNominalValueSplitterImpl implements
    SamplesByNominalValueSplitter {

  const SamplesByNominalValueSplitterImpl();

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
}