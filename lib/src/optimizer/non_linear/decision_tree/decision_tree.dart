import 'package:ml_algo/src/optimizer/non_linear/decision_tree/best_stump_finder/best_stump_finder.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_node.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_detector/leaf_detector.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:xrange/zrange.dart';

class DecisionTreeOptimizer {
  DecisionTreeOptimizer(
      Matrix observations,
      this._featuresRanges,
      this._outcomesRange,
      this._rangeToCategoricalValues,
      this._leafDetector,
      this._bestStumpFinder,
  ) {
    _root = _createNode(observations, 0);
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
      return DecisionTreeNode.leaf();
    }

    final bestStump = _bestStumpFinder.find(observations, _outcomesRange,
        _featuresRanges, _rangeToCategoricalValues);

    final childNodes = bestStump.outputObservations.map((nodeObservations) =>
        _createNode(nodeObservations, nodesCount + 1));

    return DecisionTreeNode.fromStump(bestStump, childNodes);
  }
}
