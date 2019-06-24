import 'package:ml_algo/src/optimizer/non_linear/decision_tree/assessor/stump_assessor.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_detector/leaf_detector.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/stump_selector/stump_selector.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:xrange/zrange.dart';

class DecisionTreeNode {
  DecisionTreeNode(this.children);

  final Iterable<DecisionTreeNode> children;
}

class DecisionTreeOptimizer {
  DecisionTreeOptimizer(
      Matrix features,
      Matrix outcomes,
      this._assessor,
      this._leafDetector,
      this._stumpSelector,
      [
        this._featuresRanges,
      ]
  ) :
        _outcomesRange = ZRange.closedOpen(
            features.columnsNum,
            features.columnsNum + outcomes.columnsNum
        ) {
    _root = _createTree(Matrix.fromColumns([
      ...features.columns,
      ...outcomes.columns,
    ]), 0);
  }

  final StumpAssessor _assessor;
  final LeafDetector _leafDetector;
  final StumpSelector _stumpSelector;
  final Iterable<ZRange> _featuresRanges;
  final ZRange _outcomesRange;
  DecisionTreeNode _root;

  /// Builds a tree, where each node is a logical rule, that divides given data
  /// into several parts
  DecisionTreeNode _createTree(Matrix observations, int nodesCount) {
    if (_leafDetector.isLeaf(observations, _outcomesRange, nodesCount)) {
      return DecisionTreeNode([]);
    }
    final range = _findSplittingFeatureRange(observations);
    final children = _stumpSelector.select(observations, range,
        _outcomesRange)
        .map((selected) => _createTree(selected, nodesCount + 1));
    return DecisionTreeNode(children);
  }

  ZRange _findSplittingFeatureRange(Matrix observations) {
    final errors = <double, List<ZRange>>{};
    _featuresRanges.forEach((range) {
      final stump = _stumpSelector.select(observations, range,
          _outcomesRange);
      final error = _assessor.getErrorOnStump(stump, _outcomesRange);
      errors.putIfAbsent(error, () => []);
      errors[error].add(range);
    });
    final sorted = errors.keys.toList(growable: false)
      ..sort();
    final minError = sorted.first;
    return errors[minError].first;
  }
}
