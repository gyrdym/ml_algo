import 'package:ml_algo/src/classifier/decision_tree_classifier/helpers/create_tree_svg_markup/create_tree_svg_markup_constants.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/helpers/create_tree_svg_markup/format_predicate.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/helpers/create_tree_svg_markup/get_tree_node_lines_markup.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';

String getTreeNodeMarkup(TreeNode node, num x, num y, num? childSpacing) {
  if (node.isFake) {
    return '';
  }

  final labelX = x + labelMargin;

  final splitIndex = node.splitIndex;
  final predicateType = node.predicateType;
  final splitValue = node.splitValue;
  final rectMarkup = _getRectMarkup(x, y);
  final linesMarkup = getTreeNodeLinesMarkup(node.children, x, y, childSpacing);

  if (splitIndex == null || predicateType == null || splitValue == null) {
    return _getRootNodeMarkup(x, y, linesMarkup);
  }

  final conditionLabel = formatPredicate(predicateType);
  final valueLabel = formatValue(splitValue);
  final getLabelY = (int num) => y + 15 + labelHeight * num;

  final splitIndexMarkup =
      '<text class="label" x="$labelX" y="${getLabelY(1)}">Column index</text>'
      '<text class="value" x="${labelX + labelWidth}" y="${getLabelY(1)}">${node.splitIndex}</text>';

  final predicateMarkup =
      '<text class="label" x="$labelX" y="${getLabelY(2)}">Split condition</text>'
      '<text class="value" x="${labelX + labelWidth}" y="${getLabelY(2)}">$conditionLabel $valueLabel</text>';

  return '<g>'
      '$rectMarkup'
      '$splitIndexMarkup'
      '$predicateMarkup'
      '$linesMarkup'
      '</g>';
}

String _getRootNodeMarkup(num x, num y, String linesMarkup) {
  final rectMarkup = _getRectMarkup(x, y);

  return '<g>'
      '$rectMarkup'
      '<text class="root-node-label" x="${x + 65}" y="${y + 50}">ROOT</text>'
      '$linesMarkup'
      '</g>';
}

String _getRectMarkup(num x, num y) {
  return '<rect rx="15" style="$nodeStyle" x="$x" y="$y" width="$nodeWidth" height="$nodeHeight"></rect>';
}

String formatValue(num value) {
  return value.toStringAsFixed(2);
}
