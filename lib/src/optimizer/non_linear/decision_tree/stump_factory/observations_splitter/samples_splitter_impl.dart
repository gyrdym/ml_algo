import 'package:ml_algo/src/optimizer/non_linear/decision_tree/stump_factory/observations_splitter/samples_splitter.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class SamplesSplitterImpl implements SamplesSplitter {
  const SamplesSplitterImpl();

  @override
  List<Matrix> split(Matrix samples, int splittingColumnIdx,
      double splittingValue) {
    final source1 = <Vector>[];
    final source2 = <Vector>[];
    samples.rows.forEach((row) => row[splittingColumnIdx] >= splittingValue
        ? source2.add(row)
        : source1.add(row));
    return [Matrix.fromRows(source1), Matrix.fromRows(source2)];
  }
}
