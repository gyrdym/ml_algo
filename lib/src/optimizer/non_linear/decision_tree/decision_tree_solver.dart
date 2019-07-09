import 'package:ml_algo/src/optimizer/non_linear/decision_tree/best_stump_finder/best_stump_finder.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_node.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_detector/leaf_detector.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_label_factory/leaf_label_factory.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:quiver/iterables.dart';
import 'package:xrange/zrange.dart';

import 'nominal_splitter/nominal_splitter.dart';
import 'numerical_splitter/numerical_splitter.dart';

class DecisionTreeSolver {
  DecisionTreeSolver(
      Matrix samples,
      this._featuresColumnRanges,
      this._outcomeColumnRange,
      this._rangeToNominalValues,
      this._leafDetector,
      this._leafLabelFactory,
      this._bestStumpFinder,
      this._nominalSplitter,
      this._numericalSplitter,
  ) : _isOutcomeNominal = _rangeToNominalValues
      .containsKey(_outcomeColumnRange) {
    _root = _createNode(samples, _featuresColumnRanges);
  }

  final Iterable<ZRange> _featuresColumnRanges;
  final ZRange _outcomeColumnRange;
  final Map<ZRange, List<Vector>> _rangeToNominalValues;
  final bool _isOutcomeNominal;
  final LeafDetector _leafDetector;
  final DecisionTreeLeafLabelFactory _leafLabelFactory;
  final BestStumpFinder _bestStumpFinder;
  final NumericalSplitter _numericalSplitter;
  final NominalSplitter _nominalSplitter;

  DecisionTreeNode get root => _root;
  DecisionTreeNode _root;

  Map<DecisionTreeNode, Matrix> traverse(Matrix samples) =>
      _traverse(samples, _root, {});

  DecisionTreeNode _createNode(Matrix samples,
      Iterable<ZRange> featuresColumnRanges) {
    if (_leafDetector.isLeaf(samples, _outcomeColumnRange,
        featuresColumnRanges)) {
      return DecisionTreeNode.leaf(_leafLabelFactory.create(
          samples,
          _outcomeColumnRange,
          _isOutcomeNominal,
      ));
    }

    final bestStump = _bestStumpFinder.find(samples, _outcomeColumnRange,
        featuresColumnRanges, _rangeToNominalValues);

    final bestSplittingRange = bestStump.splittingColumnRange;

    final isBestSplitByNominalValue = _rangeToNominalValues
        .containsKey(bestSplittingRange);

    final updatedColumnRanges = isBestSplitByNominalValue
        ? (Set<ZRange>.from(featuresColumnRanges)..remove(bestSplittingRange))
        : featuresColumnRanges;

    final childNodes = bestStump.outputSamples.map((samples) =>
        _createNode(samples, updatedColumnRanges));

    return DecisionTreeNode.fromStump(bestStump,
        childNodes.toList(growable: false));
  }

  Map<DecisionTreeNode, Matrix> _traverse(Matrix samples, DecisionTreeNode node,
      Map<DecisionTreeNode, Matrix> leafNodesToSamples) {
    if (node.isLeaf) {
      return leafNodesToSamples
        ..update(node, null, ifAbsent: () => samples);
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
        _traverse(
          indexedSplitPart.value,
          node.children[indexedSplitPart.index],
          leafNodesToSamples,
        )
    );

    return leafNodesToSamples;
  }
}
