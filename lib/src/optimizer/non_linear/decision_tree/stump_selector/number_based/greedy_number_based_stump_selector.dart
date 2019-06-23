import 'package:ml_algo/src/optimizer/non_linear/decision_tree/assessor/stump_assessor.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/stump_selector/number_based/node_splitter/node_splitter.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/stump_selector/number_based/number_based_stump_selector.dart';
import 'package:ml_linalg/axis.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:xrange/zrange.dart';

class GreedyNumberBasedStumpSelector implements NumberBasedStumpSelector {
  GreedyNumberBasedStumpSelector(this._assessor, this._nodeSplitter);

  final StumpAssessor _assessor;
  final NodeSplitter _nodeSplitter;

  @override
  List<Matrix> select(Matrix observations, int selectedColumnIdx,
      ZRange outcomesRange) {
    final errors = <double, List<Matrix>>{};
    final sortedRows = observations
        .sort((row) => row[selectedColumnIdx], Axis.rows).rows;
    var prevValue = sortedRows.first[selectedColumnIdx];
    for (final row in sortedRows.skip(1)) {
      final splittingValue = (prevValue + row[selectedColumnIdx]) / 2;
      final stump = _nodeSplitter
          .split(observations, selectedColumnIdx, splittingValue);
      final error = _assessor.getErrorOnStump(stump, outcomesRange);
      errors[error] = stump;
      prevValue = row[selectedColumnIdx];
    }
    final sorted = errors.keys.toList(growable: false)
      ..sort();
    final minError = sorted.first;
    return errors[minError];
  }
}
