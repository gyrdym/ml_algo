import 'package:ml_algo/src/solver/non_linear/decision_tree/decision_tree_leaf_label.dart';
import 'package:ml_algo/src/solver/non_linear/decision_tree/decision_tree_node.dart';
import 'package:ml_algo/src/solver/non_linear/decision_tree/leaf_detector/leaf_detector.dart';
import 'package:ml_algo/src/solver/non_linear/decision_tree/leaf_label_factory/leaf_label_factory.dart';
import 'package:ml_algo/src/solver/non_linear/decision_tree/split_selector/split_selector.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:xrange/zrange.dart';

class DecisionTreeSolver {
  DecisionTreeSolver(
      Matrix samples,
      this._featuresColumnIdxs,
      this._targetIdx,
      this._colIdxToUniqueValues,
      this._leafDetector,
      this._leafLabelFactory,
      this._splitSelector,
  ) : _isOutcomeNominal = _colIdxToUniqueValues
      .containsKey(_targetIdx) {
    _root = _createNode(samples, null, null, null, null,
        _featuresColumnIdxs, 0);
  }

  final Iterable<int> _featuresColumnIdxs;
  final int _targetIdx;
  final Map<int, List<double>> _colIdxToUniqueValues;
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
      double splittingNominalValue,
      int splittingIdx,
      TestSamplePredicate splittingClause,
      Iterable<int> featuresColumnIdxs,
      int depth,
  ) {
    if (_leafDetector.isLeaf(samples, _targetIdx, featuresColumnIdxs, depth)) {
      final label = _leafLabelFactory.create(
        samples,
        _targetIdx,
        _isOutcomeNominal,
      );
      return DecisionTreeNode(
        splittingClause,
        splittingNumericalValue,
        splittingNominalValue,
        splittingIdx,
        null,
        label,
      );
    }

    final bestSplit = _splitSelector.select(
        samples,
        _targetIdx,
        featuresColumnIdxs,
        _colIdxToUniqueValues,
    );

    final newDepth = depth + 1;

    final childNodes = bestSplit.entries.map((entry) {
      final splitNode = entry.key;
      final splitSamples = entry.value;

      final isSplitByNominalValue = _colIdxToUniqueValues
          .containsKey(splitNode.splittingIdx);

      final updatedColumnRanges = isSplitByNominalValue
          ? (Set<int>.from(featuresColumnIdxs)
              ..remove(splitNode.splittingIdx))
          : featuresColumnIdxs;

      return _createNode(
          splitSamples,
          splitNode.splittingNumericalValue,
          splitNode.splittingNominalValue,
          splitNode.splittingIdx,
          splitNode.testSample,
          updatedColumnRanges,
          newDepth);
    });

    return DecisionTreeNode(
        splittingClause,
        splittingNumericalValue,
        splittingNominalValue,
        splittingIdx,
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
