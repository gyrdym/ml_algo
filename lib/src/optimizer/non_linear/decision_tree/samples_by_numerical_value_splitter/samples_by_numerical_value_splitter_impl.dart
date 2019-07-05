import 'package:ml_algo/src/optimizer/non_linear/decision_tree/samples_by_numerical_value_splitter/samples_by_numerical_value_splitter.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class SamplesByNumericalValueSplitterImpl implements
    SamplesByNumericalValueSplitter {

  const SamplesByNumericalValueSplitterImpl();

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
