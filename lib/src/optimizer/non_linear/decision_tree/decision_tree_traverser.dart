import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_leaf_label.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_node.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/nominal_splitter/nominal_splitter.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/numerical_splitter/numerical_splitter.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:quiver/iterables.dart';

typedef OnLeafReachedCallbackFn = void Function(List<int> indices,
    DecisionTreeLeafLabel label);

class DecisionTreeTraverser {
  DecisionTreeTraverser(this._numericalSplitter, this._nominalSplitter);

  final NumericalSplitter _numericalSplitter;
  final NominalSplitter _nominalSplitter;

  Matrix _samples;
  OnLeafReachedCallbackFn _onLeafReached;

  void traverse(Matrix samples, DecisionTreeNode node,
      void onLeafReached(List<int> indices, DecisionTreeLeafLabel label)) {
    _samples = samples;
    _onLeafReached = onLeafReached;
    _traverse([], node);
  }

  void _traverse(List<int> indices, DecisionTreeNode node) {
    if (node.isLeaf) {
      _onLeafReached(indices, node.label);
      return;
    }

    final splitIndices = node.splittingNumericalValue != null
        ? _numericalSplitter.getSplittingIndices(
        _samples,
        node.splittingColumnRange.firstValue,
        node.splittingNumericalValue
    )
        : _nominalSplitter.getSplittingIndices(
        _samples,
        node.splittingColumnRange,
        node.splittingNominalValues
    );

    enumerate(splitIndices).forEach((indicesGroup) =>
        _traverse(
          indicesGroup.value,
          node.children[indicesGroup.index],
        )
    );
  }
}
