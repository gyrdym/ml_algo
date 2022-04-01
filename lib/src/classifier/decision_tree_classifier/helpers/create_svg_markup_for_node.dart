import 'dart:math' as math;
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';

const nodeWidth = 200;
const nodeHeight = 200;
const nodeHorizontalMargin = 20;
const nodeVerticalMargin = 50;

const labelMargin = 20;
const labelWidth = 70;
const labelHeight = 20;
const noValue = '-';

String createSvgMarkupForNode(TreeNode node) {
  final shape = node.shape;
  final widestLevelLength = shape.values.reduce(math.max);
  final totalWidth = widestLevelLength * (nodeWidth + nodeHorizontalMargin) -
      nodeHorizontalMargin;
  final totalHeight = shape.length * (nodeHeight + nodeVerticalMargin);
  final rootX = (totalWidth / 2 - nodeWidth / 2).floor();

  return '<svg width="$totalWidth" height="$totalHeight">${_traverse(node, rootX, 20)}</svg>';
}

String _traverse(TreeNode node, int x, int y) {
  return _createNodeMarkup(node, x, y);
}

String _createNodeMarkup(TreeNode node, int x, int y) {
  final labelX = x + labelMargin;
  final getLabelHeight = (int num) => y + labelHeight * num;

  return '<rect x="$x" y="$y" width="$nodeWidth" height="$nodeHeight">'
      '<text x="$labelX" y="${getLabelHeight(1)}">Value:</text>'
      '<text x="${labelX + labelWidth}" y="${getLabelHeight(1)}">${node.splittingValue ?? noValue}</text>'
      '<text x="$labelX" y="${getLabelHeight(2)}">Split index:</text>'
      '<text x="${labelX + labelWidth}" y="${getLabelHeight(2)}">${node.splittingIndex ?? noValue}</text>'
      '<text x="$labelX" y="${getLabelHeight(3)}">Split predicate:</text>'
      '<text x="${labelX + labelWidth}" y="${getLabelHeight(3)}">${node.predicateType ?? noValue}</text>'
      '</rect>';
}
