import 'package:ml_algo/src/tree_trainer/splitter/numerical_splitter/numerical_splitter.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/kd_tree_node/kd_tree_node.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/tree_node_splitting_predicate_type.dart';
import 'package:ml_linalg/matrix.dart';

class KDTreeTrainer {
  KDTreeTrainer(
    this._leafSize,
    this._splitter,
  );

  final int _leafSize;
  final NumericalTreeSplitter _splitter;

  KDTreeNode train(Matrix samples) => _train(samples, null, null, null, 0);

  KDTreeNode _train(
    Matrix samples,
    num? splittingValue,
    int? splittingIdx,
    TreeNodeSplittingPredicateType? splittingPredicateType,
    int level,
  ) {
    final isLeaf = samples.rowsNum <= _leafSize;

    if (isLeaf) {
      return KDTreeNode(
        splittingPredicateType,
        splittingValue,
        splittingIdx,
        [],
        samples,
        level,
      );
    }

    final bestSplit = _selectSplit(samples);

    final newLevel = level + 1;

    final childNodes = bestSplit.entries.map((entry) {
      final splitNode = entry.key;
      final splitSamples = entry.value;

      return _train(
        splitSamples,
        splitNode.splittingValue,
        splitNode.splittingIndex,
        splitNode.predicateType,
        newLevel,
      );
    });

    return KDTreeNode(
      splittingPredicateType,
      splittingValue,
      splittingIdx,
      childNodes.toList(growable: false),
      null,
      level,
    );
  }

  Map<KDTreeNode, Matrix> _selectSplit(Matrix samples) {
    final variances = samples.variance();

    var colIdx = 0;
    var maxIdx = colIdx;
    var max = variances[maxIdx];

    variances.forEach((variance) {
      if (variance > max) {
        max = variance;
        maxIdx = colIdx;
      }
      colIdx++;
    });

    final splittingColumn = samples.getColumn(maxIdx);

    return _splitter.split<KDTreeNode>(samples, maxIdx, splittingColumn.mean());
  }
}
