import 'package:ml_algo/src/tree_solver/tree_node.dart';
import 'package:ml_algo/src/tree_solver/leaf_detector/leaf_detector.dart';
import 'package:ml_algo/src/tree_solver/leaf_label/leaf_label.dart';
import 'package:ml_algo/src/tree_solver/leaf_label/leaf_label_factory.dart';
import 'package:ml_algo/src/tree_solver/split_selector/split_selector.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class TreeSolver {
  TreeSolver(
      Matrix samples,
      this._featureIndices,
      this._targetIdx,
      this._featureToUniqueValues,
      this._leafDetector,
      this._leafLabelFactory,
      this._splitSelector,
  ) {
    _root = _createNode(samples, null, null, null, _featureIndices, 0);
  }

  final Iterable<int> _featureIndices;
  final int _targetIdx;
  final Map<int, List<num>> _featureToUniqueValues;
  final TreeLeafDetector _leafDetector;
  final TreeLeafLabelFactory _leafLabelFactory;
  final TreeSplitSelector _splitSelector;

  TreeNode get root => _root;
  TreeNode _root;

  TreeLeafLabel getLabelForSample(Vector sample) =>
      _getLabelForSample(sample, _root);

  TreeNode _createNode(
      Matrix samples,
      num splittingValue,
      int splittingIdx,
      TestSamplePredicate splittingClause,
      Iterable<int> featuresColumnIdxs,
      int level,
  ) {
    if (_leafDetector.isLeaf(samples, _targetIdx, featuresColumnIdxs, level)) {
      final label = _leafLabelFactory.create(samples, _targetIdx);
      return TreeNode(
        splittingClause,
        splittingValue,
        splittingIdx,
        null,
        label,
        level,
      );
    }

    final bestSplit = _splitSelector.select(
        samples,
        _targetIdx,
        featuresColumnIdxs,
        _featureToUniqueValues,
    );

    final newLevel = level + 1;

    final childNodes = bestSplit.entries.map((entry) {
      final splitNode = entry.key;
      final splitSamples = entry.value;

      final isSplitByNominalValue = _featureToUniqueValues
          .containsKey(splitNode.splittingIdx);

      final updatedColumnRanges = isSplitByNominalValue
          ? (Set<int>.from(featuresColumnIdxs)
              ..remove(splitNode.splittingIdx))
          : featuresColumnIdxs;

      return _createNode(
          splitSamples,
          splitNode.splittingValue,
          splitNode.splittingIdx,
          splitNode.testSample,
          updatedColumnRanges,
          newLevel);
    });

    return TreeNode(
        splittingClause,
        splittingValue,
        splittingIdx,
        childNodes.toList(growable: false),
        null,
        level);
  }

  TreeLeafLabel _getLabelForSample(Vector sample,
      TreeNode node) {
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
