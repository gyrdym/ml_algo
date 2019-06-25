import 'package:ml_algo/src/optimizer/non_linear/decision_tree/stump_selector/observations_splitter/observations_splitter.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class ObservationsSplitterImpl implements ObservationsSplitter {
  const ObservationsSplitterImpl();

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
