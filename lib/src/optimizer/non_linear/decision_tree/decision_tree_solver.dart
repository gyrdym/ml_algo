import 'package:ml_algo/src/optimizer/non_linear/decision_tree/best_split_finder/best_split_finder.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_node.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_detector/leaf_detector.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_label_factory/leaf_label_factory.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:xrange/zrange.dart';

class DecisionTreeSolver {
  DecisionTreeSolver(
      Matrix samples,
      this._featuresColumnRanges,
      this._outcomeColumnRange,
      this._rangeToNominalValues,
      this._leafDetector,
      this._leafLabelFactory,
      this._bestStumpFinder,
  ) : _isOutcomeNominal = _rangeToNominalValues
      .containsKey(_outcomeColumnRange) {
    _root = _createNode(samples, null, null, null, null, _featuresColumnRanges);
  }

  final Iterable<ZRange> _featuresColumnRanges;
  final ZRange _outcomeColumnRange;
  final Map<ZRange, List<Vector>> _rangeToNominalValues;
  final bool _isOutcomeNominal;
  final LeafDetector _leafDetector;
  final DecisionTreeLeafLabelFactory _leafLabelFactory;
  final BestSplitFinder _bestStumpFinder;

  DecisionTreeNode get root => _root;
  DecisionTreeNode _root;

  Map<DecisionTreeNode, Matrix> traverse(Matrix samples) =>
      _traverse(samples, _root, {});

  DecisionTreeNode _createNode(
      Matrix samples,
      double splittingNumericalValue,
      Vector splittingNominalValue,
      ZRange splittingRange,
      FilterPredicate splittingClause,
      Iterable<ZRange> featuresColumnRanges,
  ) {
    if (_leafDetector.isLeaf(samples, _outcomeColumnRange,
        featuresColumnRanges)) {
      final label = _leafLabelFactory.create(
        samples,
        _outcomeColumnRange,
        _isOutcomeNominal,
      );
      return DecisionTreeNode(
        splittingClause,
        splittingNumericalValue,
        splittingNominalValue,
        splittingRange,
        null,
        label,
      );
    }

    final bestSplit = _bestStumpFinder.find(
        samples,
        _outcomeColumnRange,
        featuresColumnRanges,
        _rangeToNominalValues,
    );

    final childNodes = bestSplit.entries.map((entry) {
      final splitNode = entry.key;
      final splitSamples = entry.value;

      final isSplitByNominalValue = _rangeToNominalValues
          .containsKey(splitNode.splittingColumnRange);

      final updatedColumnRanges = isSplitByNominalValue
          ? (Set<ZRange>.from(featuresColumnRanges)
              ..remove(splitNode.splittingColumnRange))
          : featuresColumnRanges;

      return _createNode(
          splitSamples,
          splitNode.splittingNumericalValue,
          splitNode.splittingNominalValue,
          splitNode.splittingColumnRange,
          splitNode.isSampleAcceptable,
          updatedColumnRanges);
    });

    return DecisionTreeNode(
        splittingClause,
        splittingNumericalValue,
        splittingNominalValue,
        splittingRange,
        childNodes.toList(growable: false),
        null);
  }

  Map<DecisionTreeNode, Matrix> _traverse(Matrix samples, DecisionTreeNode node,
      Map<DecisionTreeNode, Matrix> leafNodesToSamples) {
    if (node.isLeaf) {
      return leafNodesToSamples
        ..update(node, null, ifAbsent: () => samples);
    }

    node.children.forEach((node) {
      final nodeSamplesSource = samples.rows.where(node.isSampleAcceptable)
          .toList();
      final nodeSamples = Matrix.fromRows(nodeSamplesSource);
      _traverse(nodeSamples, node, leafNodesToSamples);
    });

    return leafNodesToSamples;
  }
}
