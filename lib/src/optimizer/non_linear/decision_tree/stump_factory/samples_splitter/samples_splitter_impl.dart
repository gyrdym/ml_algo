import 'package:ml_algo/src/optimizer/non_linear/decision_tree/stump_factory/samples_splitter/samples_splitter.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class SamplesSplitterImpl implements SamplesSplitter {
  const SamplesSplitterImpl();

  @override
  List<Matrix> split(Matrix samples, int splittingColumnIdx,
      double splittingValue) {
    final left = <Vector>[];
    final right = <Vector>[];
    samples.rows.forEach((row) => row[splittingColumnIdx] < splittingValue
        ? left.add(row)
        : right.add(row));
    return [Matrix.fromRows(left), Matrix.fromRows(right)];
  }
}
