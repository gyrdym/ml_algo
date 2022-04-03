import 'package:ml_algo/src/classifier/decision_tree_classifier/helpers/create_tree_svg_markup/create_tree_svg_markup_constants.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/helpers/create_tree_svg_markup/get_tree_levels.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/helpers/create_tree_svg_markup/get_tree_node_distance_by_level.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/helpers/create_tree_svg_markup/get_tree_node_markup.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/helpers/create_tree_svg_markup/get_tree_width.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';

class _NodesMarkupData {
  _NodesMarkupData(
      {required this.markup, required this.level, required this.y});

  final String markup;
  final int level;
  final num y;
}

String createTreeSvgMarkup(TreeNode node) {
  final shape = node.shape;
  final levels = getTreeLevels(node, shape.length);
  final nodeDistanceByLevel =
      getTreeNodeDistanceByLevel(levels, nodeWidth, minNodeHorizontalDistance);
  final totalWidth = getTreeWidth(levels, nodeWidth, minNodeHorizontalDistance);
  final totalHeight = shape.length * (nodeHeight + nodeVerticalDistance);
  final markup = _generateMarkup(levels, node, nodeDistanceByLevel);

  return '<svg xmlns="http://www.w3.org/2000/svg" width="$totalWidth" height="$totalHeight">$textStyles$markup</svg>';
}

String _generateMarkup(
    List<List<TreeNode>> levels, TreeNode root, Map<int, num> distByLevel) {
  return levels.fold<_NodesMarkupData>(
      _NodesMarkupData(markup: '', level: 0, y: 20), (data, nodes) {
    final spacing = distByLevel[data.level]!;
    final childSpacing = distByLevel.containsKey(data.level + 1)
        ? distByLevel[data.level + 1]!
        : null;
    final getX =
        (int idx) => spacing / 2 + (idx == 0 ? 0 : idx * (nodeWidth + spacing));

    var nodeIdx = 0;
    final nodesMarkup = nodes
        .map((node) =>
            getTreeNodeMarkup(node, getX(nodeIdx++), data.y, childSpacing))
        .join();

    return _NodesMarkupData(
      markup: '${data.markup}$nodesMarkup',
      level: data.level + 1,
      y: data.y + nodeHeight + nodeVerticalDistance,
    );
  }).markup;
}
