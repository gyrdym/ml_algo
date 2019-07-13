import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_leaf_label.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_node.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_detector/leaf_detector.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_label_factory/leaf_label_factory.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/split_selector/split_selector.dart';
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
      this._splitSelector,
  ) : _isOutcomeNominal = _rangeToNominalValues
      .containsKey(_outcomeColumnRange) {
    _root = _createNode(samples, null, null, null, null,
        _featuresColumnRanges, 0);
  }

  final Iterable<ZRange> _featuresColumnRanges;
  final ZRange _outcomeColumnRange;
  final Map<ZRange, List<Vector>> _rangeToNominalValues;
  final bool _isOutcomeNominal;
  final LeafDetector _leafDetector;
  final DecisionTreeLeafLabelFactory _leafLabelFactory;
  final SplitSelector _splitSelector;

  DecisionTreeNode get root => _root;
  DecisionTreeNode _root;

  DecisionTreeLeafLabel getLabelForSample(Vector sample) =>
      _getLabelForSample(sample, _root);

  DecisionTreeNode _createNode(
      Matrix samples,
      double splittingNumericalValue,
      Vector splittingNominalValue,
      ZRange splittingRange,
      TestSamplePredicate splittingClause,
      Iterable<ZRange> featuresColumnRanges,
      int depth,
  ) {
    if (_leafDetector.isLeaf(samples, _outcomeColumnRange,
        featuresColumnRanges, depth)) {
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

    final bestSplit = _splitSelector.select(
        samples,
        _outcomeColumnRange,
        featuresColumnRanges,
        _rangeToNominalValues,
    );

    final newDepth = depth + 1;

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
          splitNode.testSample,
          updatedColumnRanges,
          newDepth);
    });

    return DecisionTreeNode(
        splittingClause,
        splittingNumericalValue,
        splittingNominalValue,
        splittingRange,
        childNodes.toList(growable: false),
        null);
  }

  DecisionTreeLeafLabel _getLabelForSample(Vector sample,
      DecisionTreeNode node) {
    if (node.isLeaf) {
      return node.label;
    }

    for (final child in node.children) {
      if (child.testSample(sample)) {
        return _getLabelForSample(sample, child);
      }
    };

    throw Exception('The given sample does not conform any splitting '
        'condition');
  }
}
