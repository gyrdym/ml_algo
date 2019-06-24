import 'package:ml_algo/src/optimizer/non_linear/decision_tree/assessor/stump_assessor.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/best_stump_finder/best_stump_finder.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/stump_selector/stump_selector.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:xrange/zrange.dart';

class GreedyStumpFinder implements BestStumpFinder {
  GreedyStumpFinder(this._assessor, this._stumpSelector);

  final StumpAssessor _assessor;
  final StumpSelector _stumpSelector;

  @override
  Iterable<Matrix> find(Matrix observations, ZRange outcomesRange,
      Iterable<ZRange> featuresRanges,
      [Map<ZRange, List<Vector>> rangeToCategoricalValues]) {
    final errors = <double, List<Iterable<Matrix>>>{};
    featuresRanges.forEach((range) {
      final categoricalValues = rangeToCategoricalValues != null
          ? rangeToCategoricalValues[range]
          : null;
      final stump = _stumpSelector.select(observations, range,
          outcomesRange, categoricalValues);
      final error = _assessor.getErrorOnStump(stump, outcomesRange);
      errors.update(error, (stumps) => stumps..add(stump),
          ifAbsent: () => [stump]);
    });
    final sorted = errors.keys.toList(growable: false)
      ..sort();
    final minError = sorted.first;
    return errors[minError].first;
  }
}
