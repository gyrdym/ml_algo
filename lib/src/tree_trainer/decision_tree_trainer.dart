import 'package:ml_algo/src/tree_trainer/leaf_detector/leaf_detector.dart';
import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label_factory.dart';
import 'package:ml_algo/src/tree_trainer/split_selector/split_selector.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/split_predicate/predicate_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';
import 'package:ml_algo/src/tree_trainer/tree_trainer.dart';
import 'package:ml_linalg/matrix.dart';

class DecisionTreeTrainer implements TreeTrainer {
  DecisionTreeTrainer(
    this._featureIndices,
    this._targetIdx,
    this._featureToUniqueValues,
    this._leafDetector,
    this._leafLabelFactory,
    this._splitSelector,
  );

  final Iterable<int> _featureIndices;
  final int _targetIdx;
  final Map<int, List<num>> _featureToUniqueValues;
  final TreeLeafDetector _leafDetector;
  final TreeLeafLabelFactory _leafLabelFactory;
  final TreeSplitSelector _splitSelector;

  @override
  TreeNode train(Matrix samples) =>
      _train(samples, null, null, null, _featureIndices, 0);

  TreeNode _train(
    Matrix samples,
    num? splittingValue,
    int? splittingIdx,
    PredicateType? splittingPredicateType,
    Iterable<int> featuresColumnIdxs,
    int level,
  ) {
    final isLeaf = _leafDetector.isLeaf(
      samples,
      _targetIdx,
      featuresColumnIdxs,
      level,
    );

    if (isLeaf) {
      final label = _leafLabelFactory.create(
        samples,
        _targetIdx,
      );

      return TreeNode(
        splittingPredicateType,
        splittingValue,
        splittingIdx,
        null,
        label,
      );
    }

    final bestSplit = _splitSelector.select(
      samples,
      _targetIdx,
      featuresColumnIdxs,
      _featureToUniqueValues,
    );

    final childNodes = bestSplit.entries.map((entry) {
      final splitNode = entry.key;
      final splitSamples = entry.value;
      final isSplitByNominalValue =
          _featureToUniqueValues.containsKey(splitNode.splitIndex);
      final updatedColumnRanges = isSplitByNominalValue
          ? (Set<int>.from(featuresColumnIdxs)..remove(splitNode.splitIndex))
          : featuresColumnIdxs;

      return _train(
        splitSamples,
        splitNode.splitValue,
        splitNode.splitIndex,
        splitNode.predicateType,
        updatedColumnRanges,
        level + 1,
      );
    });

    return TreeNode(
      splittingPredicateType,
      splittingValue,
      splittingIdx,
      childNodes.toList(growable: false),
      null,
    );
  }
}
