import 'dart:math' as math;
import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/tree_node_splitting_predicate_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';

const nodeWidth = 150;
const nodeHeight = 120;
const nodeHorizontalMargin = 20;
const nodeVerticalMargin = 50;
const nodeStyle = 'fill:blue;fill-opacity:.8';

const labelMargin = 20;
const labelWidth = 100;
const labelHeight = 20;
const noValue = '-';

String createSvgMarkupForNode(TreeNode node) {
  final shape = node.shape;
  final widestLevelLength = shape.values.reduce(math.max);
  final totalWidth = widestLevelLength * (nodeWidth + nodeHorizontalMargin);
  final totalHeight = shape.length * (nodeHeight + nodeVerticalMargin);
  final rootX = (totalWidth / 2 - nodeWidth / 2).floor();

  return '<svg xmlns="http://www.w3.org/2000/svg" width="$totalWidth" height="$totalHeight">${_traverse(node, rootX, 20)}</svg>';
}

String _traverse(TreeNode node, int x, int y) {
  final children = node.children;

  if (children == null) {
    return _createNodeMarkup(node, x, y);
  }

  final childY = y + nodeHeight + nodeVerticalMargin;
  final startChildX = (x + nodeWidth ~/ 2) -
      (children.length ~/ 2) * (nodeWidth + nodeHorizontalMargin);
  final getChildX = (int idx) =>
      startChildX + idx == 0 ? 0 : (idx * nodeWidth + nodeHorizontalMargin);
  final nodeMarkup = _createNodeMarkup(node, x, y);

  var childIdx = 0;
  final childMarkup = children
      .map((child) => _traverse(child, getChildX(childIdx++), childY))
      .join();

  return '$nodeMarkup$childMarkup';
}

String _createNodeMarkup(TreeNode node, int x, int y) {
  final labelX = x + labelMargin;
  final getLabelHeight = (int num) => y + labelHeight * num;

  final valueMarkup = '<text x="$labelX" y="${getLabelHeight(1)}">Value:</text>'
      '<text x="${labelX + labelWidth}" y="${getLabelHeight(1)}">${formatValue(node.splittingValue)}</text>';

  final splitIndexMarkup =
      '<text x="$labelX" y="${getLabelHeight(2)}">Split index:</text>'
      '<text x="${labelX + labelWidth}" y="${getLabelHeight(2)}">${formatSplitIndex(node.splittingIndex)}</text>';

  final predicateMarkup =
      '<text x="$labelX" y="${getLabelHeight(3)}">Split predicate:</text>'
      '<text x="${labelX + labelWidth}" y="${getLabelHeight(3)}">${formatPredicate(node.predicateType)}</text>';

  return '<g>'
      '<rect rx="10" style="$nodeStyle" x="$x" y="$y" width="$nodeWidth" height="$nodeHeight"></rect>'
      '$valueMarkup'
      '$splitIndexMarkup'
      '$predicateMarkup'
      '</g>';
}

String formatValue(num? value) {
  return value?.toStringAsFixed(2) ?? noValue;
}

String formatSplitIndex(int? index) {
  return index?.toString() ?? noValue;
}

String formatPredicate(TreeNodeSplittingPredicateType? predicate) {
  switch (predicate) {
    case TreeNodeSplittingPredicateType.lessThan:
      return '&#60;';

    case TreeNodeSplittingPredicateType.lessThanOrEqualTo:
      return '&#8804;';

    case TreeNodeSplittingPredicateType.equalTo:
      return '==';

    case TreeNodeSplittingPredicateType.greaterThan:
      return '&#62;';

    case TreeNodeSplittingPredicateType.greaterThanOrEqualTo:
      return '&#8805;';

    default:
      return noValue;
  }
}
