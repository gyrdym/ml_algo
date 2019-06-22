import 'package:ml_algo/src/optimizer/non_linear/decision_tree/assessor/stump_assessor.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/stump_selector/number_based/node_splitter/node_splitter.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/stump_selector/number_based/number_based_stump_selector.dart';
import 'package:ml_linalg/axis.dart';
import 'package:ml_linalg/matrix.dart';

class GreedyNumberBasedStumpSelector implements NumberBasedStumpSelector {
  const GreedyNumberBasedStumpSelector(this._assessor, this._nodeSplitter);

  final StumpAssessor _assessor;
  final NodeSplitter _nodeSplitter;

  @override
  List<Matrix> select(Matrix observations, int index) {
    final errors = <double, List<Matrix>>{};
    final rows = observations.sort((row) => row[index], Axis.rows).rows;
    var prevValue = rows.first[index];
    for (final row in rows.skip(1)) {
      final splittingValue = (prevValue + row[index]) / 2;
      final stump = _nodeSplitter.split(observations, index, splittingValue);
      final error = _assessor.getErrorOnStump(stump);
      errors[error] = stump;
      prevValue = row[index];
    }
    final sorted = errors.keys.toList(growable: false)
      ..sort();
    final minError = sorted.first;
    return errors[minError];
  }
}
