import 'package:ml_algo/src/optimizer/non_linear/decision_tree/best_stump_finder/best_stump_finder.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_node.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_detector/leaf_detector.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_label_factory/leaf_label_factory.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:xrange/zrange.dart';

class DecisionTreeOptimizer<T> {
  DecisionTreeOptimizer(
      Matrix observations,
      this._featuresRanges,
      this._outcomesColumnRange,
      this._rangeToCategoricalValues,
      this._leafDetector,
      this._leafLabelFactory,
      this._bestStumpFinder,
  ) {
    _root = _createNode(observations, 0);
  }

  final Iterable<ZRange> _featuresRanges;
  final Map<ZRange, List<Vector>> _rangeToCategoricalValues;
  final LeafDetector _leafDetector;
  final DecisionTreeLeafLabelFactory _leafLabelFactory;
  final BestStumpFinder _bestStumpFinder;
  final ZRange _outcomesColumnRange;
  DecisionTreeNode _root;

  /// Builds a tree, where each node is a logical rule, that divides given data
  /// into several parts
  DecisionTreeNode _createNode(Matrix observations, int nodesCount) {
    if (_leafDetector.isLeaf(observations, _outcomesColumnRange, nodesCount)) {
      return DecisionTreeNode.leaf(_leafLabelFactory.create(
          observations,
          _outcomesColumnRange,
          _rangeToCategoricalValues.containsKey(_outcomesColumnRange)),
      );
    }

    final bestStump = _bestStumpFinder.find(observations, _outcomesColumnRange,
        _featuresRanges, _rangeToCategoricalValues);

    final childNodes = bestStump.outputObservations.map((nodeObservations) =>
        _createNode(nodeObservations, nodesCount + 1));

    return DecisionTreeNode.fromStump(bestStump, childNodes);
  }
}
