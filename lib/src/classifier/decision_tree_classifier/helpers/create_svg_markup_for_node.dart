import 'package:ml_algo/src/classifier/decision_tree_classifier/helpers/get_dist_between_nodes.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/helpers/get_tree_levels.dart';
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
  final levels = getTreeLevels(node, shape.length);
  final distData = getDistBetweenNodes(levels, nodeWidth, nodeHorizontalMargin);
  final totalWidth = distData.totalWidth;
  final totalHeight = shape.length * (nodeHeight + nodeVerticalMargin);
  final rootX = (totalWidth / 2 - nodeWidth / 2).floor();

  return '<svg xmlns="http://www.w3.org/2000/svg" width="$totalWidth" height="$totalHeight">${_traverse(levels, node, distData.distByLevel)}</svg>';
}

String _traverse(
    List<List<TreeNode>> levels, TreeNode node, Map<int, num> distByLevel) {
  return levels.fold<Map<String, dynamic>>({'markup': '', 'y': 20, 'level': 0},
      (data, nodes) {
    final markup = data['markup'] as String;
    final level = data['level'] as int;
    final y = data['y'] as num;
    final childSpacing = distByLevel[level]!;
    final getX = (int idx) =>
        childSpacing / 2 + (idx == 0 ? 0 : idx * (nodeWidth + childSpacing));

    var childIdx = 0;
    final childMarkup = nodes
        .map((child) => _createNodeMarkup(child, getX(childIdx++), y))
        .join();

    return {
      'markup': '$markup$childMarkup',
      'y': y + nodeHeight + nodeVerticalMargin,
      'level': level + 1,
    };
  })['markup'] as String;
}

String _createNodeMarkup(TreeNode node, num x, num y) {
  if (node.isFake) {
    return '';
  }

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
