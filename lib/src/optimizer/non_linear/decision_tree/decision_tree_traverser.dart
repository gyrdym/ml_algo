import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_leaf_label.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_node.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/nominal_splitter/nominal_splitter.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/numerical_splitter/numerical_splitter.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:quiver/iterables.dart';

typedef OnLeafReachedCallbackFn = void Function(Matrix samples,
    DecisionTreeLeafLabel label);

class DecisionTreeTraverser {
  DecisionTreeTraverser(this._numericalSplitter, this._nominalSplitter);

  final NumericalSplitter _numericalSplitter;
  final NominalSplitter _nominalSplitter;

  void traverse(Matrix samples, DecisionTreeNode node,
      OnLeafReachedCallbackFn onLeafReached) {
    if (node.isLeaf) {
      onLeafReached(samples, node.label);
      return;
    }

    final split = node.splittingNumericalValue != null
        ? _numericalSplitter.split(
        samples,
        node.splittingColumnRange.firstValue,
        node.splittingNumericalValue
    )
        : _nominalSplitter.split(
        samples,
        node.splittingColumnRange,
        node.splittingNominalValues
    );

    enumerate(split).forEach((indexedSplitPart) =>
        traverse(
          indexedSplitPart.value,
          node.children[indexedSplitPart.index],
          onLeafReached,
        )
    );
  }
}
