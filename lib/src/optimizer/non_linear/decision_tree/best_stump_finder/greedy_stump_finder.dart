import 'package:ml_algo/src/optimizer/non_linear/decision_tree/best_stump_finder/best_stump_finder.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_stump.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/split_assessor/split_assessor.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/stump_factory/stump_factory.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:xrange/zrange.dart';

class GreedyStumpFinder implements BestStumpFinder {
  GreedyStumpFinder(this._assessor, this._stumpFactory);

  final SplitAssessor _assessor;
  final StumpFactory _stumpFactory;

  @override
  DecisionTreeStump find(Matrix samples, ZRange outcomesColumnRange,
      Iterable<ZRange> featuresColumnRanges,
      [Map<ZRange, List<Vector>> rangeToNominalValues]) {
    final errors = <double, List<DecisionTreeStump>>{};
    featuresColumnRanges.forEach((range) {
      final nominalValues = rangeToNominalValues != null
          ? rangeToNominalValues[range]
          : null;
      final stump = _stumpFactory.create(samples, range,
          outcomesColumnRange, nominalValues);
      final error = _assessor.getAggregatedError(stump.outputSamples,
          outcomesColumnRange);
      errors.update(error, (stumps) => stumps..add(stump),
          ifAbsent: () => [stump]);
    });
    final sorted = errors.keys.toList(growable: false)
      ..sort();
    final minError = sorted.first;
    return errors[minError].first;
  }
}
