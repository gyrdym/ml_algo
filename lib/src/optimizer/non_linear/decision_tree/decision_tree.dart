import 'package:ml_algo/src/optimizer/non_linear/decision_tree/best_stump_finder/best_stump_finder.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_node.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_detector/leaf_detector.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_label_factory/leaf_label_factory.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/stump_factory/samples_numerical_splitter/samples_numerical_splitter.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:quiver/iterables.dart';
import 'package:xrange/zrange.dart';

class DecisionTreeOptimizer {
  DecisionTreeOptimizer(
      Matrix samples,
      this._featuresColumnRanges,
      this._outcomeColumnRange,
      this._rangeToNominalValues,
      this._leafDetector,
      this._leafLabelFactory,
      this._bestStumpFinder,
      this._samplesSplitter,
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
  final SamplesNumericalSplitter _samplesSplitter;

  DecisionTreeNode get root => _root;
  DecisionTreeNode _root;

  void traverse(Matrix samples, DecisionTreeNode node) {
    if (node.isLeaf) {
      print(node.label);
      return;
    }
    if (node.splittingNumericalValue != null) {
      zip([
        _samplesSplitter.split(
            samples,
            node.splittingColumnRange.firstValue,
            node.splittingNumericalValue
        ),
        node.children,
      ])
      .forEach((pair) => traverse(
          pair.first as Matrix,
          pair.last as DecisionTreeNode,
      ));
      return;
    }
  }

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
}
