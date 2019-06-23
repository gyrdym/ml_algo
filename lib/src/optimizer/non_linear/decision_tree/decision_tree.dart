import 'package:ml_algo/src/optimizer/non_linear/decision_tree/assessor/stump_assessor.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_detector/leaf_detector.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/stump_selector/number_based/number_based_stump_selector.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/stump_selector/vector_based/vector_based_stump_selector.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
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
      this._numberBasedStumpSelector,
      this._vectorBasedStumpSelector,
      [
        this._featuresRanges,
      ]
  ) :
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
  final NumberBasedStumpSelector _numberBasedStumpSelector;
  final VectorBasedStumpSelector _vectorBasedStumpSelector;
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
          ? _vectorBasedStumpSelector.select(observations, target,
          categoricalValues)
          : _numberBasedStumpSelector.select(observations, target.firstValue);
}
