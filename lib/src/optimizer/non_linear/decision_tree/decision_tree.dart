import 'package:ml_algo/src/optimizer/non_linear/decision_tree/assessor/stump_assessor.dart';
import 'package:ml_linalg/axis.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:xrange/zrange.dart';

import 'leaf_detector/leaf_detector.dart';
import 'node_splitter/node_splitter.dart';

class DecisionTreeNode {
  DecisionTreeNode(this.children);

  final Iterable<DecisionTreeNode> children;
}

class DecisionTreeOptimizer {
  DecisionTreeOptimizer(Matrix features, Matrix outcomes, this._assessor,
      this._leafDetector, this._numberBasedNodeSplitter,
      [this._featuresRanges]) :
        _outcomesRange = ZRange.closedOpen(
            features.columnsNum,
            features.columnsNum + outcomes.columnsNum
        ) {
    _root = _createNode(Matrix.fromColumns([
      ...features.columns,
      ...outcomes.columns,
    ]), 0);
  }

  final StumpAssessor _assessor;
  final LeafDetector _leafDetector;
  final NodeSplitter _numberBasedNodeSplitter;
  final Iterable<ZRange> _featuresRanges;
  final ZRange _outcomesRange;
  DecisionTreeNode _root;

  /// Builds a tree, where each node is a logical rule, that divides given data
  /// into several parts
  DecisionTreeNode _createNode(Matrix observations, int nodesCount) {
    final outcomes = observations.submatrix(columns: _outcomesRange);
    if (_leafDetector.isLeaf(outcomes, nodesCount)) {
      return DecisionTreeNode([]);
    }
    final range = _findSplittingFeatureRange(observations);
    final children = _learnStump(observations, range)
        .map((selected) => _createNode(selected, nodesCount + 1));
    return DecisionTreeNode(children);
  }

  ZRange _findSplittingFeatureRange(Matrix observations) {
    final errors = <double, List<ZRange>>{};
    _featuresRanges.forEach((range) {
      final stump = _learnStump(observations, range);
      final error = _assessor.getErrorOnStump(stump);
      errors.putIfAbsent(error, () => []);
      errors[error].add(range);
    });
    final sorted = errors.keys.toList(growable: false)
      ..sort();
    final minError = sorted.first;
    return errors[minError].first;
  }

  Iterable<Matrix> _learnStump(Matrix observations, ZRange target,
      [List<Vector> categoricalValues]) =>
      categoricalValues?.isNotEmpty == true
          ? _getObservationsByCategoricalValues(observations, target,
          categoricalValues)
          : _getObservationsByRealValue(observations, target.firstValue);

  List<Matrix> _getObservationsByCategoricalValues(Matrix observations,
      ZRange range, List<Vector> splittingValues) =>
    splittingValues.map((value) {
      final foundRows = observations.rows
          .where((row) => row.subvectorByRange(range) == value)
          .toList(growable: false);
      return Matrix.fromRows(foundRows);
    }).toList(growable: false);

  List<Matrix> _getObservationsByRealValue(Matrix observations, int index) {
    final errors = <double, List<Matrix>>{};
    final rows = observations.sort((row) => row[index], Axis.rows).rows;
    var prevValue = rows.first[index];
    for (final row in rows.skip(1)) {
      final splittingValue = (prevValue + row[index]) / 2;
      final stump = _numberBasedNodeSplitter
          .split(observations, index, splittingValue);
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
