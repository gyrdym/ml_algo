import 'package:ml_algo/src/tree_trainer/splitter/nominal_splitter/nominal_splitter.dart';
import 'package:ml_algo/src/tree_trainer/splitter/numerical_splitter/numerical_splitter.dart';
import 'package:ml_algo/src/tree_trainer/splitter/splitter.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';
import 'package:ml_linalg/axis.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:xrange/xrange.dart';

class GreedyTreeSplitter implements TreeSplitter {
  GreedyTreeSplitter(
      this._assessor, this._numericalSplitter, this._nominalSplitter);

  final TreeAssessor _assessor;
  final NumericalTreeSplitter _numericalSplitter;
  final NominalTreeSplitter _nominalSplitter;

  @override
  Map<TreeNode, Matrix> split(Matrix samples, int splittingIdx, int targetId,
          [List<num>? uniqueValues]) =>
      uniqueValues != null
          ? _createByNominalValues(samples, splittingIdx, uniqueValues)
          : _createByNumericalValue(samples, splittingIdx, targetId);

  Map<TreeNode, Matrix> _createByNominalValues(
      Matrix samples, int splittingIdx, List<num> values) {
    if (splittingIdx < 0 || splittingIdx > samples.columnCount) {
      throw Exception('Inappropriate range given: $splittingIdx, '
          'expected a range within or equal '
          '${integers(0, samples.columnCount)}');
    }
    return _nominalSplitter.split(samples, splittingIdx, values);
  }

  Map<TreeNode, Matrix> _createByNumericalValue(
      Matrix samples, int splittingIdx, int targetId) {
    final errors = <double, List<Map<TreeNode, Matrix>>>{};
    final sortedRows =
        samples.sort((row) => row[splittingIdx], Axis.rows).rows.toList();
    final rowsWithoutFirst = sortedRows.skip(1).toList();
    var prevValue = sortedRows.first[splittingIdx];

    for (var i = 0; i < rowsWithoutFirst.length; i++) {
      final row = rowsWithoutFirst[i];
      final nextValue = row[splittingIdx];

      if (prevValue == nextValue && i < sortedRows.length - 2) {
        continue;
      }

      final splittingValue = (prevValue + nextValue) / 2;
      final split =
          _numericalSplitter.split(samples, splittingIdx, splittingValue);
      final error = _assessor.getAggregatedError(split.values, targetId);

      errors.update(error, (splits) => splits..add(split),
          ifAbsent: () => [split]);

      prevValue = row[splittingIdx];
    }

    final sorted = errors.keys.toList(growable: false)..sort();
    final minError = sorted.first;

    return errors[minError]!.first;
  }
}
