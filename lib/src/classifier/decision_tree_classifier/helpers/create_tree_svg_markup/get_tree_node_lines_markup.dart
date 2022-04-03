import 'package:ml_algo/src/classifier/decision_tree_classifier/helpers/create_tree_svg_markup/create_tree_svg_markup_constants.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';

String getTreeNodeLinesMarkup(
    List<TreeNode>? children, num x, num y, num? childSpacing) {
  if (children == null || childSpacing == null) {
    return '';
  }

  final x1 = x + nodeWidth / 2;
  final y1 = y + nodeHeight;
  final startX2 =
      x1 - (children.length / 2) * (nodeWidth / 2 + childSpacing / 2);
  final getX2 =
      (int idx) => startX2 + (idx == 0 ? 0 : idx * (nodeWidth + childSpacing));
  final y2 = y + nodeHeight + nodeVerticalDistance;

  var idx = 0;
  return children.map((node) {
    return '<line x1="$x1" y1="$y1" x2="${getX2(idx++)}" y2="$y2" style="$nodeLineStyle" />';
  }).join('');
}
