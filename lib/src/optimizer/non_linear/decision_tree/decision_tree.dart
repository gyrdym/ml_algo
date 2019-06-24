import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_detector/leaf_detector.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/best_stump_finder/best_stump_finder.dart';
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
      this._featuresRanges,
      this._rangeToCategoricalValues,
      this._leafDetector,
      this._bestStumpFinder,
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

  final Iterable<ZRange> _featuresRanges;
  final Map<ZRange, List<Vector>> _rangeToCategoricalValues;
  final LeafDetector _leafDetector;
  final BestStumpFinder _bestStumpFinder;
  final ZRange _outcomesRange;
  DecisionTreeNode _root;

  /// Builds a tree, where each node is a logical rule, that divides given data
  /// into several parts
  DecisionTreeNode _createNode(Matrix observations, int nodesCount) {
    if (_leafDetector.isLeaf(observations, _outcomesRange, nodesCount)) {
      return DecisionTreeNode([]);
    }

    final bestStump = _bestStumpFinder.find(observations, _outcomesRange,
        _featuresRanges, _rangeToCategoricalValues);

    final childNodes = bestStump.map((nodeObservations) =>
        _createNode(nodeObservations, nodesCount + 1));

    return DecisionTreeNode(childNodes);
  }
}
