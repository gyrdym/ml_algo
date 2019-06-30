import 'package:ml_algo/src/optimizer/non_linear/decision_tree/best_stump_finder/best_stump_finder.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_node.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_detector/leaf_detector.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_label_factory/leaf_label_factory.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:xrange/zrange.dart';

class DecisionTreeOptimizer {
  DecisionTreeOptimizer(
      Matrix observations,
      this._featuresColumnRanges,
      this._outcomesColumnRange,
      this._rangeToCategoricalValues,
      this._leafDetector,
      this._leafLabelFactory,
      this._bestStumpFinder,
  ) {
    _root = _createNode(observations);
  }

  final Iterable<ZRange> _featuresColumnRanges;
  final ZRange _outcomesColumnRange;
  final Map<ZRange, List<Vector>> _rangeToCategoricalValues;
  final LeafDetector _leafDetector;
  final DecisionTreeLeafLabelFactory _leafLabelFactory;
  final BestStumpFinder _bestStumpFinder;

  DecisionTreeNode get root => _root;
  DecisionTreeNode _root;

  DecisionTreeNode _createNode(Matrix observations) {
    if (_leafDetector.isLeaf(observations, _outcomesColumnRange)) {
      return DecisionTreeNode.leaf(_leafLabelFactory.create(
          observations,
          _outcomesColumnRange,
          _rangeToCategoricalValues.containsKey(_outcomesColumnRange)),
      );
    }

    final bestStump = _bestStumpFinder.find(observations, _outcomesColumnRange,
        _featuresColumnRanges, _rangeToCategoricalValues);

    final childNodes = bestStump.outputObservations.map(_createNode);

    return DecisionTreeNode.fromStump(bestStump,
        childNodes.toList(growable: false));
  }
}
