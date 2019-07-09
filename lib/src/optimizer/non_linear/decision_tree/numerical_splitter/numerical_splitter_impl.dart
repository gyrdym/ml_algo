import 'package:ml_algo/src/optimizer/non_linear/decision_tree/numerical_splitter/numerical_splitter.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:quiver/iterables.dart';

class NumericalSplitterImpl implements NumericalSplitter {
  const NumericalSplitterImpl();

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
